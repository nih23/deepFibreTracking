"This module contains the Reinforcement Learning Environment for Tractography"
import gym
from gym import spaces
import numpy as np

from dipy.data import get_sphere
import torch


from dfibert.data.postprocessing import resample
from dfibert.data import HCPDataContainer, PointOutsideOfDWIError
from dfibert.tracker import StreamlinesFromFileTracker
from dfibert.util import get_grid

from ._state import TractographyState



class EnvTractography(gym.Env):
    "The tractography gym environment"
    def __init__(self, device, step_width = 1, dataset = '100307', grid_dim = None,
                 max_l2dist_to_terminal_state = 0.1,
                 reference_streamlines_path = "data/HCP307200_DTI_smallSet.vtk"):
        if grid_dim is None:
            grid_dim = [3,3,3]
        #data/HCP307200_DTI_min40.vtk => 5k streamlines
        print("Loading precomputed streamlines (%s) for ID %s" %
              (reference_streamlines_path, dataset))
        self.device = device
        self.dataset = HCPDataContainer(dataset)
        self.dataset.normalize() #normalize HCP data

        self.step_width = step_width
        self.dtype = torch.FloatTensor
        sphere = get_sphere("repulsion100")
        self.directions = sphere.vertices
        no_actions, _ = self.directions.shape
        self.action_space = spaces.Discrete(no_actions+1)#spaces.Discrete(no_actions)
        self.dwi_postprocessor = resample(sphere=sphere)
        self.reference_streamline_ijk = None
        self.grid = get_grid(np.array(grid_dim))
        self.max_l2dist_to_terminal_state = max_l2dist_to_terminal_state
        self.reference_streamlines_path = reference_streamlines_path

        self.state = self.reset()

        self.step_counter = 0
        self.max_steps = 200

    def interpolate_dwi_at_state(self, state_coordinates):
        "Interpolates the DWI values for the given state"
        #TODO: maybe stay in RAS all the time then no need to transfer to IJK
        ras_points = self.dataset.to_ras(state_coordinates)
        # Transform state to World RAS+ coordinate system

        ras_points = self.grid + ras_points

        try:
            interpolated_dwi = self.dataset.get_interpolated_dwi(ras_points,
                                       postprocessing=self.dwi_postprocessor)
        except PointOutsideOfDWIError as _:
            return None
        interpolated_dwi = np.rollaxis(interpolated_dwi,3) #CxWxHxD
        #interpolated_dwi = self.dtype(interpolated_dwi).to(self.device)
        return interpolated_dwi

    def step(self, action):
        if(action == (self.action_space.n - 1)) or (self.step_counter > self.max_steps):
            #print("Entering terminal state")
            done = True
            reward = self.reward_for_terminal_state(self.state)
            if reward > 0.95:
                reward += (1/self.step_counter)
            else:
                reward -= self.step_counter / (self.max_steps / 10.)
            return self.state, reward, done

        ## convert discrete action into tangent vector
        action_vector = self.directions[action]

        ## apply step by step length and update state accordingly
        next_state_position = self.state.get_coordinate() + self.step_width * action_vector
        next_state = TractographyState(next_state_position, self.interpolate_dwi_at_state)
        if next_state.get_value() is None:
            return self.state, -10, True

        ## compute reward for new state
        reward_next_state = self.reward_for_state(next_state)

        ### check if we already left brain map
        # => RLenv.dataset.data.binarymask.shape
        # set done = True if coordinate of next_state is outside of binarymask
        done = False
        self.step_counter += 1
        try:
            next_state.get_value()
        except PointOutsideOfDWIError:
            done = True
            #print("Agent left brain mask :(")
            return self.state, -10, done


        self.state = TractographyState(next_state_position, self.interpolate_dwi_at_state)
        # return step information
        return next_state, reward_next_state, done


    def reward_for_state(self, state):
        "Returns the reward for a given state"
        # In general, the reward will be negative but very close to zero if the agent is
        # staying close to our reference streamline.
        # Right now, this function only returns negative rewards but simply adding some threshold
        # to the LeakyReLU is gonna result in positive rewards, too
        #
        # We will be normalising the distance wrt. to LeakyRelu activation function.
        qry_pt = torch.FloatTensor(state.get_coordinate()).view(-1,3)
        distance = (torch.min(torch.sum( (self.reference_streamline_ijk - qry_pt)**2, dim =1 ) +
                    torch.sum( (self.reference_streamline_ijk[-1,:] - qry_pt)**4 )))
        return torch.tanh(-distance+5.3) + self.reward_for_terminal_state(state) / 2

    def reward_for_terminal_state(self, state):
        "Returns the reward for a given terminal state"
        qry_pt = torch.FloatTensor(state.get_coordinate()).view(3)
        distance = torch.sum( (self.reference_streamline_ijk[-1,:] - qry_pt)**2 )
        #return torch.where(distance < self.max_l2dist_to_terminal_state, 1, 0 )
        return torch.tanh(-distance+5.3)


    # reset the game and returns the observed data from the last episode
    def reset(self):
        file_sl = StreamlinesFromFileTracker(self.reference_streamlines_path)
        file_sl.track()

        tracked_streamlines = file_sl.get_streamlines()
        streamline_index = np.random.randint(len(tracked_streamlines))
        #print("Reset to streamline %d/%d" % (streamline_index+1, len(tracked_streamlines)))
        reference_streamline_ras = tracked_streamlines[streamline_index]
        reference_streamline_ijk = self.dataset.to_ijk(reference_streamline_ras)
        initial_position_ijk = reference_streamline_ijk[0]

        self.state = TractographyState(initial_position_ijk, self.interpolate_dwi_at_state)
        self.done = False
        self.reference_streamline_ijk = self.dtype(reference_streamline_ijk).to(self.device)

        self.step_counter = 0

        return self.state


    def render(self, mode="human"):
        pass

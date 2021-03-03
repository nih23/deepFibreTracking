import os, sys

import gym
from gym import spaces
from gym.spaces import Discrete, Box
import numpy as np

from dipy.data import get_sphere
import torch


from dfibert.data.postprocessing import res100, resample
from dfibert.data import HCPDataContainer, ISMRMDataContainer, PointOutsideOfDWIError
from dfibert.tracker import StreamlinesFromFileTracker
from dfibert.util import get_grid

from ._state import TractographyState



class RLtractEnvironment(gym.Env):
    def __init__(self, device, stepWidth = 1, dataset = '100307', grid_dim = [3,3,3], maxL2dist_to_terminalState = 0.1, pReferenceStreamlines = "data/HCP307200_DTI_smallSet.vtk"):
        #data/HCP307200_DTI_min40.vtk => 5k streamlines
        print("Loading precomputed streamlines (%s) for ID %s" % (pReferenceStreamlines, dataset))
        self.device = device
        self.dataset = HCPDataContainer(dataset)
        self.dataset.normalize() #normalize HCP data
        
        self.stepWidth = stepWidth
        self.dtype = torch.FloatTensor
        sphere = get_sphere("repulsion100")
        self.directions = sphere.vertices
        noActions, _ = self.directions.shape
        self.action_space = spaces.Discrete(noActions)#spaces.Discrete(noActions+1)
        self.dwi_postprocessor = resample(sphere=sphere)
        self.referenceStreamline_ijk = None
        self.grid = get_grid(np.array(grid_dim))
        self.maxL2dist_to_terminalState = maxL2dist_to_terminalState
        self.pReferenceStreamlines = pReferenceStreamlines
        
        self.state = self.reset()
            
        
    def interpolateDWIatState(self, stateCoordinates):       
        #TODO: maybe stay in RAS all the time then no need to transfer to IJK
        ras_points = self.dataset.to_ras(stateCoordinates) # Transform state to World RAS+ coordinate system
        
        ras_points = self.grid + ras_points
        
        interpolated_dwi = self.dataset.get_interpolated_dwi(ras_points, postprocessing=self.dwi_postprocessor)
        interpolated_dwi = np.rollaxis(interpolated_dwi,3) #CxWxHxD
        #interpolated_dwi = self.dtype(interpolated_dwi).to(self.device)
        return interpolated_dwi
  
    
    def step(self, action):  
        #if(action == (self.action_space.n - 1)):
        #    print("Entering terminal state")
        #    done = True
        #    reward = self.rewardForTerminalState(self.state)
        #    return self.state, reward, done
            
        ## convert discrete action into tangent vector
        action_vector = self.directions[action]
        
        ## apply step by step length and update state accordingly
        positionNextState = self.state.getCoordinate() + self.stepWidth * action_vector
        nextState = TractographyState(positionNextState, self.interpolateDWIatState)
        #self.state = nextState
        
        ## compute reward for new state
        rewardNextState = self.rewardForState(nextState)
        
        ### check if we already left brain map
        # => RLenv.dataset.data.binarymask.shape
        # set done = True if coordinate of nextState is outside of binarymask
        #done = is_done()
        done = False
        try:
            nextState.getValue()
        except PointOutsideOfDWIError:
            done = True
            #print("Agent left brain mask :(")
            return self.state.getValue(), -100, done

        
        self.state = TractographyState(positionNextState, self.interpolateDWIatState)
        # return step information
        return nextState.getValue(), rewardNextState, done
    
    
    def rewardForState(self, state):
        # In general, the reward will be negative but very close to zero if the agent is 
        # staying close to our reference streamline.
        # Right now, this function only returns negative rewards but simply adding some threshold 
        # to the LeakyReLU is gonna result in positive rewards, too
        #
        # We will be normalising the distance wrt. to LeakyRelu activation function. 
        qry_pt = torch.FloatTensor(state.getCoordinate()).view(-1,3)
        distance = torch.min(torch.sum( (self.referenceStreamline_ijk - qry_pt)**2, dim =1 ))
        reward = torch.nn.functional.leaky_relu(-1 * distance)
        return reward
 

    def rewardForTerminalState(self, state):
        qry_pt = torch.FloatTensor(self.state.getCoordinate()).view(3)
        distance = torch.sum( (self.referenceStreamline_ijk[-1,:] - qry_pt)**2 )
        return torch.where(distance < self.maxL2dist_to_terminalState, 1, 0 )


    # reset the game and returns the observed data from the last episode
    def reset(self):       
        file_sl = StreamlinesFromFileTracker(self.pReferenceStreamlines)
        file_sl.track()
        
        tracked_streamlines = file_sl.get_streamlines()
        streamline_index = np.random.randint(len(tracked_streamlines))
        #print("Reset to streamline %d/%d" % (streamline_index+1, len(tracked_streamlines)))
        referenceStreamline_ras = tracked_streamlines[streamline_index]
        referenceStreamline_ijk = self.dataset.to_ijk(referenceStreamline_ras)
        initialPosition_ijk = referenceStreamline_ijk[0]
        
        self.state = TractographyState(initialPosition_ijk, self.interpolateDWIatState)
        #self.state = initialPosition_ijk
        self.done = False
        self.referenceStreamline_ijk = self.dtype(referenceStreamline_ijk).to(self.device)
        
        return self.state.getValue()


    def render(self, mode="human"):
        pass

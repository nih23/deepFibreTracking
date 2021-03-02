# the framework to follow
import gym
from gym import spaces
from gym.spaces import Discrete, Box
import numpy as np
from dfibert.tracker.nn. import rl
from dfibert.data import HCPDataContainer, ISMRMDataContainer
hcp_data = HCPDataContainer(100307)
ismrm_data = ISMRMDataContainer()


class RLenv(gym.Env):
    # justifying action and observation space
    def __init__(self, device):
        self.device = device
        # 30 directions we can take
        self.action_space = spaces.Discrete(30)
        self.observation_space = spaces.Box(np.array([0,0,0]), np.array([x,y,z]))
        self.state = np.random.rand(0,3,size=3)
        self.done = Flase

    # tranform the cartesian coordinate state to the interpolated_dwi as the input of network
    def get_input_state(self.state):
        ras_points = hcp_data.to_ras(self.state) # Transform state to World RAS+ coordinate system
        interpolated_dwi = hcp_data.get_interpolated_dwi(ras_points, ignore_outside_points=False)
        input_state = interpolated_dwi

    # calculate the angle between action vector and DTI_direction_vector
    def Angle_calculator(action_vector, DTI_direction_vector):
        m=action_vector
        n=DTI_direction_vector
        l_m=np.sqrt(m.dot(m))
        l_n=np.sqrt(n.dot(n))
        dot_product=x.dot(y)
        cos_=dot_product/(l_m*l_n)
        angle=np.arccos(cos_)
        return angle

    # action will be performed and returns calculated state and reward
    def step(self, action):
        # apply action
        state += step_width * norm(action_vector)

        # calculate reward
        if angle(action_vector, DTI_direction_vector)in range(0,pi/2):
            reward = 1
        elif: angle(action_vector, DTI_direction_vector)in range(pi/2,pi):
            reward = -1
        else: angle(action_vector, DTI_direction_vector)in range(pi, 2pi):
            reward = -2

        # check if episode is done
        if self.state.is_out:
            done = True
        else:
            done = False

        # return step information
        return self.state, reward, done

    # reset the game and returns the observed data from the last episode
    def reset(self):
        # reset state
        self.state = np.random.rand(0,3,size=3)
        return self.state

    def close(self):
        self.env.close()

    # show or render an episode
    def render(self, mode="human")

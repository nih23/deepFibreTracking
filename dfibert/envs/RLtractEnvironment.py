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
        self.action_space = spaces.Discrete(noActions+1) # spaces.Discrete(noActions)
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
        
        try:
            interpolated_dwi = self.dataset.get_interpolated_dwi(ras_points, postprocessing=self.dwi_postprocessor)
        except:
            return None
        interpolated_dwi = np.rollaxis(interpolated_dwi,3) #CxWxHxD
        #interpolated_dwi = self.dtype(interpolated_dwi).to(self.device)
        return interpolated_dwi
  
    
    def step(self, action):
        self.reward -= 0.1
        self.stepCounter += 1
        if self.stepCounter > self.maxSteps:
            sphere_dist = ((self.state.getCoordinate()[0] - self.referenceStreamline_ijk[np.min([self.stepCounter, 86])][0])**2 \
                      + (self.state.getCoordinate()[1] - self.referenceStreamline_ijk[np.min([self.stepCounter, 86])][1])**2 \
                      + (self.state.getCoordinate()[2] - self.referenceStreamline_ijk[np.min([self.stepCounter, 86])][2])**2)

            if sphere_dist <= 0.2**2:
                print("Episode ended at terminal state but took too long")
                return self.state, 50, True
            else:
                return self.state, -100, True
            #reward = self.rewardForTerminalState(self.state)
            #if reward > 1:
            #    print("Episode ended at terminal state but took too long")
            #    return self.state, 40, True
            #else:
            #    return self.state, -100, True

        if action == (self.action_space.n - 1):
            #reward = self.rewardForState(self.state)
            #self.stepCounter += 1
            #return self.state, reward, False
            nextState = self.state
        else:

            ## convert discrete action into tangent vector
            action_vector = self.directions[action]
            
            ## apply step by step length and update state accordingly
            positionNextState = self.state.getCoordinate() + self.stepWidth * action_vector
            nextState = TractographyState(positionNextState, self.interpolateDWIatState)
            if nextState.getValue() is None:
                return self.state, -100, True
        
        ## compute reward for new state


        ### check if we already left brain map
        # => RLenv.dataset.data.binarymask.shape
        # set done = True if coordinate of nextState is outside of binarymask
        done = False
        try:
            nextState.getValue()
        except PointOutsideOfDWIError:
            print("PointOutside still occured")
            done = True
            #print("Agent left brain mask :(")
            return self.state, -100, done

        #rewardNextState = self.rewardForState(nextState)
        sphere_dist = ((nextState.getCoordinate()[0] - self.referenceStreamline_ijk[np.min([self.stepCounter, 86])][0])**2 \
                      + (nextState.getCoordinate()[1] - self.referenceStreamline_ijk[np.min([self.stepCounter, 86])][1])**2 \
                      + (nextState.getCoordinate()[2] - self.referenceStreamline_ijk[np.min([self.stepCounter, 86])][2])**2)

        if sphere_dist <= 0.4**2:
            self.reward += 1000/len(self.referenceStreamline_ijk)
            self.points_visited += 1
        
        rewardNextState = self.reward - self.past_reward
        self.past_reward = self.reward

        if self.points_visited == len(self.referenceStreamline_ijk):
            print("Hey, hey, hey we are finally there! :D")
            done = True
            rewardNextState = 100

        #self.state = TractographyState(positionNextState, self.interpolateDWIatState)
        self.state = nextState
        # return step information
        return nextState, rewardNextState, done
    
    
    def rewardForState(self, state):
        # In general, the reward will be negative but very close to zero if the agent is 
        # staying close to our reference streamline.
        # Right now, this function only returns negative rewards but simply adding some threshold 
        # to the LeakyReLU is gonna result in positive rewards, too
        #
        # We will be normalising the distance wrt. to LeakyRelu activation function.
        #print(state.getCoordinate())
        #qry_pt = state.getCoordinate().view(-1,3)
        #distance = torch.min(torch.sum( (self.referenceStreamline_ijk[np.max([self.stepCounter-1-2,0]):np.min([self.stepCounter-1+1,self.maxSteps])] - qry_pt)**2, dim =1 ))
        #distance = torch.min(torch.sum((self.referenceStreamline_ijk[np.min([self.stepCounter+1, self.maxSteps-1])] - qry_pt)**2, dim=1))
        
        distance = lineseg_dist(state.getCoordinate().numpy(), self.referenceStreamline_ijk[np.min([self.stepCounter, len(self.referenceStreamline_ijk)-3])].numpy(), self.referenceStreamline_ijk[np.min([self.stepCounter+2, len(self.referenceStreamline_ijk)-1])].numpy())
        #reward = 2 + (-distance/10)
        #reward = 1 + (1-distance)
        #if distance == -1:
        #    reward = 0.5
        #elif distance < 0.8:
        #    reward = 1+ (1-distance)
        #else:
        #    reward = np.max([1 - distance, -1])
       
        #return reward
        return -distance
 

    def rewardForTerminalState(self, state):
        qry_pt = state.getCoordinate().view(3)
        distance = torch.max((self.referenceStreamline_ijk[-1,:] - qry_pt)**2 )
        #return torch.where(distance < self.maxL2dist_to_terminalState, 1, 0 )
        #return torch.tanh(-distance+5.3)
        #print(distance)
        if distance < 0.5:
            reward =  2 + (distance/10)
        else:
            reward = np.max([1 - distance, -1])
        return reward


    # reset the game and returns the observed data from the last episode
    def reset(self):       
        file_sl = StreamlinesFromFileTracker(self.pReferenceStreamlines)
        file_sl.track()
        
        tracked_streamlines = file_sl.get_streamlines()
        #streamline_index = np.random.randint(len(tracked_streamlines))
        #print("Reset to streamline %d/%d" % (streamline_index+1, len(tracked_streamlines)))
        referenceStreamline_ras = tracked_streamlines[0]#tracked_streamlines[streamline_index]
        referenceStreamline_ijk = self.dataset.to_ijk(referenceStreamline_ras)
        referenceStreamline_ijk = self.dtype(referenceStreamline_ijk).to(self.device)
        initialPosition_ijk = referenceStreamline_ijk[0]
        
        self.state = TractographyState(initialPosition_ijk, self.interpolateDWIatState)
        #print(self.state.getCoordinate())
        self.done = False
        #self.referenceStreamline_ijk = self.dtype(referenceStreamline_ijk).to(self.device)
        self.referenceStreamline_ijk = referenceStreamline_ijk

        self.stepCounter = 0
        self.maxSteps = 200#len(self.referenceStreamline_ijk)

        self.reward = 0
        self.past_reward = 0
        self.points_visited = 0
        
        return self.state


    def render(self, mode="human"):
        pass
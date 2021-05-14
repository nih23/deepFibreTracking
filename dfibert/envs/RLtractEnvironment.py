import os, sys

import gym
from gym import spaces
from gym.spaces import Discrete, Box
import numpy as np

from dipy.data import get_sphere
from dipy.data import HemiSphere, Sphere
import torch


from dfibert.data.postprocessing import res100, resample
from dfibert.data import HCPDataContainer, ISMRMDataContainer, PointOutsideOfDWIError
from dfibert.tracker import StreamlinesFromFileTracker
from dfibert.util import get_grid

from ._state import TractographyState



class RLtractEnvironment(gym.Env):
    def __init__(self, device, stepWidth = 1, action_space=20, dataset = '100307', grid_dim = [3,3,3], maxL2dist_to_terminalState = 0.1, pReferenceStreamlines = "data/HCP307200_DTI_smallSet.vtk"):
        #data/HCP307200_DTI_min40.vtk => 5k streamlines
        print("Loading precomputed streamlines (%s) for ID %s" % (pReferenceStreamlines, dataset))
        self.device = device
        self.dataset = HCPDataContainer(dataset)
        self.dataset.normalize() #normalize HCP data
        
        self.stepWidth = stepWidth
        self.dtype = torch.FloatTensor
        #sphere = HemiSphere(xyz=res)#get_sphere("repulsion100")
        n_pts = action_space - 1                                    ## added action_space variable for flexible debugging
        theta = np.pi * np.random.rand(n_pts)
        phi = 2 * np.pi * np.random.rand(n_pts)
        sphere = Sphere(theta=theta, phi=phi)
        self.directions = sphere.vertices
        self.directions = np.concatenate((self.directions, np.array([[0.0, 0.0, 0.0]])))
        #self.directions = sphere.vertices #res
        noActions, _ = self.directions.shape
        #noActions = len(self.directions)
        self.action_space = spaces.Discrete(noActions) #spaces.Discrete(noActions+1)
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
            #print("Point outside of brain mask :(")
            return None
        interpolated_dwi = np.rollaxis(interpolated_dwi,3) #CxWxHxD
        #interpolated_dwi = self.dtype(interpolated_dwi).to(self.device)
        return interpolated_dwi
  
    

    def step(self, action):
        self.stepCounter += 1
        if self.stepCounter >= self.maxSteps:
            distTerminal = self.rewardForTerminalState(self.state)
            if distTerminal < 0.05:
                return self.state, 1., True#, {}
            else:
                return self.state, -distTerminal, True#, {}
        done = False
        if action == (self.action_space.n - 1):
            distTerminal = self.rewardForTerminalState(self.state)
            if distTerminal < 0.05:
                print("Defi stopped at/close to the terminal state!")
                return self.state, 1., True#, {}
            nextState = self.state
        #else:
        action_vector = self.directions[action]
        positionNextState = self.state.getCoordinate() + self.stepWidth * action_vector
        #positionNextState = self.state.getCoordinate() + action_vector
        #positionNextState = self.state.getCoordinate() + action    # <- for continous action space
        nextState = TractographyState(positionNextState, self.interpolateDWIatState)
        if nextState.getValue() is None:
            rewardNextState = self.rewardForTerminalState(nextState)
            return  self.state, -rewardNextState, True#, {}
        
        
        rewardNextState = - self.rewardForState(nextState)#self.cosineSimReward(self.state, nextState)
        #distTerminal = self.rewardForTerminalState(self.state)
        #if distTerminal < 2.5:
        #    print("Defi stopped at/close to the terminal state!")
        #    return self.state, 1., True#, {}
        
        self.state = nextState
        # return step information
        return self.state, rewardNextState, done#, {}
    
    
    def cosineSimReward(self, state, nextState):
        current_index = np.min([self.stepCounter,len(self.referenceStreamline_ijk)-1])
        path_vector = (nextState.getCoordinate() - state.getCoordinate()).squeeze(0)
        reference_vector = self.referenceStreamline_ijk[current_index]-self.referenceStreamline_ijk[current_index-1]
        cosine_sim = torch.nn.functional.cosine_similarity(path_vector, reference_vector, dim=0)
        #dist = torch.sum((self.referenceStreamline_ijk[current_index] - nextState.getCoordinate())**2)
        #dist = torch.dist()
        #if dist < 2.5:
        #    dist = 0
        #else:
        #    dist = dist - 2.5
        return cosine_sim#-dist
    # def step(self, action):
    #     #self.reward -= 0.1
    #     self.stepCounter += 1
    #     rewardNextState = 0

    #     if action == (self.action_space.n - 1):
    #         #reward = self.rewardForState(self.state)
    #         #self.stepCounter += 1
    #         #return self.state, reward, False
    #         #print(action)
    #         nextState = self.state
    #         dist_to_terminal = self.rewardForTerminalState(nextState)
    #         if dist_to_terminal < 0.1:
    #             print("Hey, hey, hey we finally stopped at the terminal state! :D")
    #             return nextState, 1, True
    #     else:
    #         ## convert discrete action into tangent vector
    #         action_vector = self.directions[action]
            
    #         ## apply step by step length and update state accordingly
    #         positionNextState = self.state.getCoordinate() + self.stepWidth * action_vector
    #         nextState = TractographyState(positionNextState, self.interpolateDWIatState)
    #         if nextState.getValue() is None:
    #             rewardNextState = self.rewardForTerminalState(nextState)
    #             return self.state, -rewardNextState, True
        
    #     ## compute reward for new state


    #     ### check if we already left brain map
    #     # => RLenv.dataset.data.binarymask.shape
    #     # set done = True if coordinate of nextState is outside of binarymask
    #     done = False
    #     # try:
    #     #     nextState.getValue()
    #     # except PointOutsideOfDWIError:
    #     #     print("PointOutside still occured")
    #     #     done = True
    #     #     #print("Agent left brain mask :(")
    #     #     return self.state, -100, done

    #     rewardNextState = self.rewardForState(nextState)
    #     if rewardNextState < -np.exp(2*(0.81 )-1):
    #       done = True
    #     # if rewardNextState < 0.:
    #     #     done = True
    #     if self.stepCounter > self.maxSteps:
    #         #if rewardNextState > 0.:
    #         #    rewardNextState = 50    
    #         done = True       
        
    #     # if self.points_visited == len(self.referenceStreamline_ijk):
    #     #     print("Hey, hey, hey we finally visited all tiles! :D")
    #     #     done = True
    #     #     #rewardNextState = 100

    #     self.state = nextState
    #     # return step information
    #     return nextState, rewardNextState, done
    

    
    def rewardForState(self, state):
        # In general, the reward will be negative but very close to zero if the agent is 
        # staying close to our reference streamline.
        # Right now, this function only returns negative rewards but simply adding some threshold 
        # to the LeakyReLU is gonna result in positive rewards, too
        #
        # We will be normalising the distance wrt. to LeakyRelu activation function.
        #print(state.getCoordinate())
        current_index = np.min([self.stepCounter,len(self.referenceStreamline_ijk)-1])
        qry_pt = state.getCoordinate().view(-1,3)
        l2_distance = torch.sum((self.referenceStreamline_ijk[current_index] - qry_pt)**2)

        #if l2_distance < 0.2:
        #    l2_distance = 0.
        #else:
        #    l2_distance = np.abs(0.2-l2_distance)
        #rewardNextState = -torch.exp(2*l2_distance-1)
        #distance = torch.sum((self.referenceStreamline_ijk[current_index] - qry_pt)**2)
        
        
        # if distance > 1:
        #     #print("Point outside sphere tresh of 2.25:", sphere_dist)
        #     rewardNextState = -10#-= 1000.0 / len(self.referenceStreamline_ijk)

        # if distance <= 0.1:
        #     self.reward += 1000.0 / len(self.referenceStreamline_ijk)
        #     self.points_visited += 1
        #     #print("Point currently in", sphere_dist)
        
        # rewardNextState = self.reward - self.past_reward
        # self.past_reward = self.reward
        
        return l2_distance#rewardNextState
 

    def rewardForTerminalState(self, state):
        qry_pt = state.getCoordinate().view(3)
        distance = torch.sum((self.referenceStreamline_ijk[-1,:] - qry_pt)**2)
        #return torch.where(distance < self.maxL2dist_to_terminalState, 1, 0 )
        #return torch.tanh(-distance+5.3)
        #print(distance)
        #if distance < 0.5:
        #    reward =  2 + (distance/10)
        #else:
        #    reward = np.max([1 - distance, -1])
        #return reward
        return distance

    # reset the game and returns the observed data from the last episode
    def reset(self, streamline_index=None):       
        file_sl = StreamlinesFromFileTracker(self.pReferenceStreamlines)
        file_sl.track()
        
        tracked_streamlines = file_sl.get_streamlines()
        if streamline_index == None:
            streamline_index = np.random.randint(len(tracked_streamlines))
        #print("Reset to streamline %d/%d" % (streamline_index+1, len(tracked_streamlines)))
        referenceStreamline_ras = tracked_streamlines[streamline_index]
        referenceStreamline_ijk = self.dataset.to_ijk(referenceStreamline_ras)
        referenceStreamline_ijk = self.dtype(referenceStreamline_ijk).to(self.device)
        
        #position_index = np.random.randint(len(referenceStreamline_ijk)-10)
        initialPosition_ijk = referenceStreamline_ijk[0]
        
        self.done = False
        #self.referenceStreamline_ijk = self.dtype(referenceStreamline_ijk).to(self.device)
        self.referenceStreamline_ijk = referenceStreamline_ijk
        self.state = TractographyState(initialPosition_ijk, self.interpolateDWIatState)

        self.stepCounter = 0
        self.maxSteps = 200#len(self.referenceStreamline_ijk)

        self.reward = 0
        self.past_reward = 0
        self.points_visited = 0#position_index
        
        return self.state


    def render(self, mode="human"):
        pass
import os, sys

import gym
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

import shapely.geometry as geom

from collections import deque

class RLtractEnvironment(gym.Env):
    def __init__(self, device, stepWidth = 1, action_space=20, dataset = '100307', grid_dim = [3,3,3], maxL2dist_to_State = 0.1, pReferenceStreamlines = "data/HCP307200_DTI_smallSet.vtk"):
        #data/HCP307200_DTI_min40.vtk => 5k streamlines
        print("Loading precomputed streamlines (%s) for ID %s" % (pReferenceStreamlines, dataset))
        self.device = device
        self.dataset = HCPDataContainer(dataset)
        self.dataset.normalize() #normalize HCP data
        
        self.stepWidth = stepWidth
        self.dtype = torch.FloatTensor
        #sphere = HemiSphere(xyz=res)#get_sphere("repulsion100")
        n_pts = action_space - 1                                    ## added action_space variable for flexible debugging
        np.random.seed(42)
        theta = np.pi * np.random.rand(n_pts)
        phi = 2 * np.pi * np.random.rand(n_pts)
        sphere = HemiSphere(theta=theta, phi=phi)#Sphere(theta=theta, phi=phi)
        self.directions = sphere.vertices
        self.directions = np.concatenate((self.directions, np.array([[0.0, 0.0, 0.0]])))
        #self.directions = sphere.vertices #res
        noActions, _ = self.directions.shape
        #noActions = len(self.directions)

        self.action_space = Discrete(noActions) #spaces.Discrete(noActions+1)
        #self.observation_space = Box(low=0, high=150, shape=(2700,))
        self.dwi_postprocessor = resample(sphere=get_sphere('repulsion100'))    #resample(sphere=sphere)
        self.referenceStreamline_ijk = None
        self.grid = get_grid(np.array(grid_dim))
        self.maxL2dist_to_State = maxL2dist_to_State
        self.pReferenceStreamlines = pReferenceStreamlines

        self.maxSteps = 2000
        
        #self.state = self.reset()
        self.reset()
        
        obs_shape = self.getObservationFromState(self.state).shape
        
        self.observation_space = Box(low=0, high=150, shape=obs_shape)

        self.max_mean_step_angle = 1.5
        self.max_step_angle = 1.59

        
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
        done = False
        if self.stepCounter >= self.maxSteps:
            done = True
        
        distTerminal = self.rewardForTerminalState(self.state)
        if distTerminal < self.maxL2dist_to_State:
                print("Defi reached the terminal state!")
                return self.state, 1., True, {}    
        # if action == (self.action_space.n - 1):
        #     distTerminal = self.rewardForTerminalState(self.state)
        #     if distTerminal < self.maxL2dist_to_State:
        #         print("Defi stopped at/close to the terminal state!")
        #         return self.state, 1., True, {}
        #    nextState = self.state
        #else:
        action_vector = self.directions[action]
        action_vector = self._correct_direction(action_vector)

        positionNextState = self.state.getCoordinate() + self.stepWidth * action_vector
        #positionNextState = self.state.getCoordinate() + action    # <- for continous action space
        nextState = TractographyState(positionNextState, self.interpolateDWIatState)
        if nextState.getValue() is None:
            #rewardNextState = self.rewardForTerminalState(nextState)
            print("Defi left brain mask")
            return  self.state, -100., True, {}

        
        self.state_history.append(nextState)

        #check if angle between actions is too large
        # step_cosine_similarity = 1.
        # if self.stepCounter > 1:
        #     past_path = list(self.state_history)[2] - list(self.state_history)[1]  ### todo
        #     current_path = nextState.getCoordinate() - self.state.getCoordinate()
        #     #print("past path: ", past_path)
        #     #print("current_path: ", current_path)
        #     step_cosine_similarity = self.cosineSimilarity(past_path, current_path)
        #     #print("step cosine sim: ", step_cosine_similarity)
        #     step_angle = self.arccos(step_cosine_similarity)
        #     if step_angle > self.max_step_angle:
        #         print("Angle of past to current action too high!")
        #         done = True

        #     self.step_angles.append(step_angle)

        #     if np.mean(self.step_angles) > self.max_mean_step_angle:
        #         print("Mean of all step angles too high! Step counter: {}, Mean: {}".format(self.stepCounter, np.mean(self.step_angles)))
        #         done = True
        
        # # rewardNextState = self.rewardForState(nextState)
        # # if rewardNextState > 0.:
        # #     self.points_visited += 1
        # # else:
        # #     lower_index = np.min([self.points_visited+1, len(self.referenceStreamline_ijk)-1])
        # #     upper_index = np.min([self.points_visited+4, len(self.referenceStreamline_ijk)-1])
        # #     next_l2_distances = [torch.dist(self.referenceStreamline_ijk[i], nextState.getCoordinate(), p=2) for i in range(lower_index, upper_index)]
        # #     if any(distance < self.maxL2dist_to_State for distance in next_l2_distances):
        # #         rewardNextState = 1.
        # #         self.points_visited += 1
        # #         print("Defi got close to a state further down the stream line!")

        # streamline_cosine_similarity = self.cosineSimtoStreamline(self.state, nextState)
        # # #print("Cosine sim to streamline: ", streamline_cosine_similarity)
        # rewardNextState = streamline_cosine_similarity * step_cosine_similarity

        # #rewardNextState = 1 - self.lineDistance(nextState)
        
        current_index = np.min([self.closestStreamlinePoint(self.state) + 1, len(self.referenceStreamline_ijk)-1])
        rewardNextState = self.rewardForState(nextState)
        if action == (self.action_space.n - 1):
          if distTerminal < self.maxL2dist_to_State:
              rewardNextState = -1.
        
        self.state = nextState
        #self.state_history.append(nextState)
        # return step information
        return nextState, rewardNextState, done, {}

    def _correct_direction(self, action_vector):
        # handle keeping the agent to go in the direction we want
        if self.stepCounter <= 1:                                                               # if no past states
            last_direction = self.referenceStreamline_ijk[1] - self.referenceStreamline_ijk[0]  # get the direction in which the reference steamline is going
        else:                                                                                   # else get the direction the agent has been going so far
            last_direction = self.state_history[-1].getCoordinate() - self.state_history[-2].getCoordinate()

        if np.dot(action_vector, last_direction) < 0:                                           # if the agent chooses the complete opposite direction
            action_vector = -1 * action_vector                                                  # force it to follow the rough direction of the streamline
        return action_vector

    def _get_best_action(self):
        distances = []
        current_index = np.min([self.closestStreamlinePoint(self.state) + 1, len(self.referenceStreamline_ijk)-1])
        # current_index = np.min([self.points_visited, len(self.referenceStreamline_ijk)-1]
        #if self.rewardForTerminalState(self.state) < self.maxL2dist_to_State:
        #    return self.action_space.n - 1
        
        for i in range(self.action_space.n):
            action_vector = self.directions[i]
            action_vector = self._correct_direction(action_vector)
            positionNextState = self.state.getCoordinate() + self.stepWidth * action_vector
            # nextState = TractographyState(positionNextState, self.interpolateDWIatState)
            # distances.append(self.lineDistance(nextState))                                               # array aus line distances, dann argmin für beste aktion

            # positionNextState = self.state.getCoordinate() + self.stepWidth * action_vector
            l2_distance = torch.dist(self.referenceStreamline_ijk[current_index], positionNextState.view(-1,3), p=2)
            distances.append(l2_distance)

        return np.argmin(distances)
    
    def cosineSimtoStreamline(self, state, nextState):
        #current_index = np.min([self.points_visited,len(self.referenceStreamline_ijk)-1])
        current_index = np.min([self.closestStreamlinePoint(self.state) + 1, len(self.referenceStreamline_ijk)-1])
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
    
    def rewardForState(self, state):
        # In general, the reward will be negative but very close to zero if the agent is 
        # staying close to our reference streamline.
        # Right now, this function only returns negative rewards but simply adding some threshold 
        # to the LeakyReLU is gonna result in positive rewards, too
        #
        # We will be normalising the distance wrt. to LeakyRelu activation function.
        #print(state.getCoordinate())
        #current_index = np.min([self.points_visited, len(self.referenceStreamline_ijk)-1])
        current_index = np.min([self.closestStreamlinePoint(self.state) + 1, len(self.referenceStreamline_ijk)-1])
        qry_pt = state.getCoordinate().view(-1,3)
        self.l2_distance = torch.dist(self.referenceStreamline_ijk[current_index], qry_pt, p=2)

        # if self.l2_distance > 2.5:
        #     rewardNextState = -1.
        # elif self.l2_distance < self.maxL2dist_to_State:
        #     rewardNextState = 1.
        # else:
        #     rewardNextState = 0.
        
        return 1 - self.l2_distance
 

    def rewardForTerminalState(self, state):
        qry_pt = state.getCoordinate().view(3)
        distance = torch.sum((self.referenceStreamline_ijk[-1,:] - qry_pt)**2)
        return distance

    def cosineSimilarity(self, path_1, path_2):
        return torch.nn.functional.cosine_similarity(path_2, path_1, dim=0)

    def arccos(self, angle):
        return torch.arccos(angle)

    def getObservationFromState(self, state):
        dwi_values = state.getValue().flatten()
        past_coordinates = np.array(list(self.state_history)).flatten()
        return np.concatenate((dwi_values, past_coordinates))

    def lineDistance(self, state):
        #point = geom.Point(state.getCoordinate())
        #return point.distance(self.line)
        point = geom.Point(self.state.getCoordinate())
        
        # our action should be closest to the optimal point on our streamline
        p_streamline = self.line.interpolate(self.stepCounter * self.stepWidth)
        
        return p_streamline.distance(point)

    def closestStreamlinePoint(self, state):
        distances = torch.cdist(torch.FloatTensor([self.line.coords[:]]), state.getCoordinate().unsqueeze(dim=0).float(), p=2,).squeeze(0)
        index = torch.argmin(distances)
        return index

    # reset the game and returns the observed data from the last episode
    def reset(self, streamline_index=None):       
        file_sl = StreamlinesFromFileTracker(self.pReferenceStreamlines)
        file_sl.track()
        
        tracked_streamlines = file_sl.get_streamlines()
        
        #print("No. streamlines: " + str(tracked_streamlines))
        
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
        
        self.reward = 0
        self.past_reward = 0
        self.points_visited = 1#position_index

        #self.state_history = []
        #self.state_history.append(self.state)

        coords = self.referenceStreamline_ijk
        self.line = geom.LineString(coords)

        self.step_angles = []

        self.state_history = deque(maxlen=4)
        while len(self.state_history) != 4:
            self.state_history.append(self.state)

        return self.state
        
        #return self.prepare_state(self.state)


    def render(self, mode="human"):
        pass

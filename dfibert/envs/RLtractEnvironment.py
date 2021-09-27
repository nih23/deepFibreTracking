import os, sys

import gym
from gym.spaces import Discrete, Box
import numpy as np

from dipy.data import get_sphere
from dipy.data import HemiSphere, Sphere
from dipy.core.sphere import disperse_charges
import torch


from dfibert.data.postprocessing import res100, resample
from dfibert.data import HCPDataContainer, ISMRMDataContainer, PointOutsideOfDWIError
from dfibert.tracker import StreamlinesFromFileTracker
from dfibert.util import get_grid

from ._state import TractographyState

import shapely.geometry as geom
from shapely.ops import nearest_points
from shapely.strtree import STRtree


from collections import deque

class RLtractEnvironment(gym.Env):
    def __init__(self, device, stepWidth = 1, action_space=20, dataset = '100307', grid_dim = [3,3,3], maxL2dist_to_State = 0.1, pReferenceStreamlines = "data/HCP307200_DTI_smallSet.vtk"):
        #data/HCP307200_DTI_min40.vtk => 5k streamlines
        print("Loading precomputed streamlines (%s) for ID %s" % (pReferenceStreamlines, dataset))
        self.device = device
        self.device_nnsearch = "cpu"
        self.dataset = HCPDataContainer(dataset)
        self.dataset.normalize() #normalize HCP data
        
        self.stepWidth = stepWidth
        self.dtype = torch.FloatTensor # vs. torch.cuda.FloatTensor
        #sphere = HemiSphere(xyz=res)#get_sphere("repulsion100")
        np.random.seed(42)
        
        phi = np.pi * np.random.rand(action_space)
        theta = 2 * np.pi * np.random.rand(action_space)
        sphere = HemiSphere(theta=theta, phi=phi)  #Sphere(theta=theta, phi=phi)
        sphere, potential = disperse_charges(sphere, 5000) # enforce uniform distribtuion of our points
        self.directions = sphere.vertices
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
        
        # grab streamlines from file
        file_sl = StreamlinesFromFileTracker(self.pReferenceStreamlines)
        file_sl.track()
        self.tracked_streamlines = file_sl.get_streamlines()
        self.tracked_streamlines_torch = [torch.FloatTensor(self.dataset.to_ijk(x)).to(self.device_nnsearch) for x in self.tracked_streamlines]

        
        #self.state = self.reset()
        self.reset()
        
        obs_shape = self.getObservationFromState(self.state).shape
        
        self.observation_space = Box(low=0, high=150, shape=obs_shape)

        self.max_mean_step_angle = 1.5
        self.max_step_angle = 1.59
        self.max_dist_to_referenceStreamline = 0.5 # => 0.5 average pixel distance
        


        # convert all streamlines of our dataset into IJK & Shapely LineString format
        self.lines = [geom.LineString(self.dataset.to_ijk(line)) for line in self.tracked_streamlines]

        # build search tree to locate nearest streamlines
        self.tree = STRtree(self.lines)

        
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
    
    
    def minDistToStreamline(self, streamline, state):
        dist_streamline = torch.mean( (torch.FloatTensor(streamline.coords[:]) - state.getCoordinate())**2, dim = 1 )
        return torch.min(dist_streamline)
    

    def step(self, action):
        self.stepCounter += 1
        done = False
        if self.stepCounter >= self.maxSteps:
            done = True
        
        #@TODO Nico: get rid of that
        distTerminal = self.rewardForTerminalState(self.state)
        if distTerminal < self.maxL2dist_to_State:
                #Defi reached the terminal state
                print('_', end='', flush=True)
                return self.state, 0., True, {}    

        action_vector = self.directions[action]
        action_vector = self._correct_direction(action_vector)

        positionNextState = self.state.getCoordinate() + self.stepWidth * action_vector
        nextState = TractographyState(positionNextState, self.interpolateDWIatState)
        
        #@TODO Nico: get rid of that
        if nextState.getValue() is None:
            #Defi left brain mask
            print('^', end='', flush=True)
            return self.state, 0., True, {}

        
        self.state_history.append(nextState)
        
        rewardNextState = self.rewardForState(nextState)
        self.state = nextState
        
        # if defi went to a state that is way to far from our reference streamline then the trajectory is not
        # informative anymore. There, we need to continue tracking on the closest streamline and penalize Defi
        # for that move.
        if(rewardNextState < (1-self.max_dist_to_referenceStreamline)):
            #print("Defi left reference streamline: switching to closest one.")
            print('#', end='', flush=True)
            
            ## locate closest streamline
            with torch.no_grad():
                dists = torch.zeros(len(self.tracked_streamlines)).to(self.device_nnsearch)
                ref_coord = nextState.getCoordinate().unsqueeze(dim=0).float().to(self.device_nnsearch)

                i=0
                for sl in self.tracked_streamlines_torch:
                    dists[i] = torch.min(torch.mean((sl - ref_coord)**2, dim = 1))
                    i = i + 1

                amin = torch.argmin(dists)
                line_nearest = self.lines[amin]

            self.switchStreamline(line_nearest)
            rewardNextState -= 0.1
        
        return nextState, rewardNextState, done, {}

    
    def _correct_direction(self, action_vector):
        # handle keeping the agent to go in the direction we want
        if self.stepCounter <= 1:                                                               # if no past states
            last_direction = self.referenceStreamline_ijk[1] - self.referenceStreamline_ijk[0]  # get the direction in which the reference steamline is going
        else: # else get the direction the agent has been going so far
            last_direction = self.state_history[-1].getCoordinate() - self.state_history[-2].getCoordinate()

        if np.dot(action_vector, last_direction) < 0: # if the agent chooses the complete opposite direction
            action_vector = -1 * action_vector  # force it to follow the rough direction of the streamline
        return action_vector

    
    def _get_best_action(self):
        with torch.no_grad():
            distances = []

            for i in range(self.action_space.n):
                action_vector = self.directions[i]
                action_vector = self._correct_direction(action_vector)
                positionNextState = self.state.getCoordinate() + self.stepWidth * action_vector

                dist_streamline = torch.mean( (torch.FloatTensor(self.line.coords[:]) - positionNextState)**2, dim = 1 )

                distances.append(torch.min(dist_streamline))

        return np.argmin(distances)

    
    
    def cosineSimtoStreamline(self, state, nextState):
        #current_index = np.min([self.points_visited,len(self.referenceStreamline_ijk)-1])
        current_index = np.min([self.closestStreamlinePoint(self.state) + 1, len(self.referenceStreamline_ijk)-1])
        path_vector = (nextState.getCoordinate() - state.getCoordinate()).squeeze(0)
        reference_vector = self.referenceStreamline_ijk[current_index]-self.referenceStreamline_ijk[current_index-1]
        cosine_sim = torch.nn.functional.cosine_similarity(path_vector, reference_vector, dim=0)
        return cosine_sim

    
    def rewardForState(self, state):
        # The reward will be close to one if the agent is 
        # staying close to our reference streamline.
        self.l2_distance = self.minDistToStreamline(self.line, state)
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

    '''
    def lineDistance(self, state):
        #point = geom.Point(state.getCoordinate())
        #return point.distance(self.line)
        point = geom.Point(self.state.getCoordinate())
        
        # our action should be closest to the optimal point on our streamline
        p_streamline = self.line.interpolate(self.stepCounter * self.stepWidth)
        
        return p_streamline.distance(point)
    '''
    
    def closestStreamlinePoint(self, state):
        distances = torch.cdist(torch.FloatTensor([self.line.coords[:]]), state.getCoordinate().unsqueeze(dim=0).float(), p=2,).squeeze(0)
        index = torch.argmin(distances)
        return index
    
    
    # switch reference streamline of environment
    ##@TODO: optimize runtime
    def switchStreamline(self, streamline):
        self.line = streamline        
        self.referenceStreamline_ijk = self.dtype(np.array(streamline.coords[:])).to(self.device)
        self.step_angles = []


    # reset the game and returns the observed data from the last episode
    def reset(self, streamline_index=None):              
        if streamline_index == None:
            streamline_index = np.random.randint(len(self.tracked_streamlines))
        referenceStreamline_ras = self.tracked_streamlines[streamline_index]
        referenceStreamline_ijk = self.dataset.to_ijk(referenceStreamline_ras)
        
        self.switchStreamline(geom.LineString(referenceStreamline_ijk))
        
        self.done = False
        self.stepCounter = 0
        self.reward = 0
        self.past_reward = 0
        self.points_visited = 1 #position_index
        
        self.state = TractographyState(self.referenceStreamline_ijk[0], self.interpolateDWIatState)
        self.state_history = deque(maxlen=4)
        while len(self.state_history) != 4:
            self.state_history.append(self.state)
        
        return self.state


    def render(self, mode="human"):
        pass

import os, sys
import torch
import numpy as np
import gym
from scipy.interpolate import RegularGridInterpolator
from gym.spaces import Discrete, Box
from collections import deque

import dipy.reconst.dti as dti
from dipy.data import HemiSphere, Sphere, get_sphere
from dipy.core.interpolation import trilinear_interpolate4d
from dipy.core.sphere import disperse_charges
from dipy.direction import peaks_from_model
from dipy.reconst.shm import order_from_ncoef, sph_harm_lookup
from dipy.tracking import utils

from dfibert.data import DataPreprocessor
from dfibert.data.postprocessing import Resample
from dfibert.util import get_grid
from ._state import TractographyState

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.csdeconv import (auto_response_ssst,
                                   mask_for_response_ssst,
                                   response_from_mask_ssst)
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

from tqdm import trange


class RLtractEnvironment(gym.Env):
    def __init__(self, device, seeds=None, stepWidth=0.8, action_space=20, dataset='100307', grid_dim=[3, 3, 3],
                 maxL2dist_to_State=0.1, tracking_in_RAS=True, fa_threshold=0.1, b_val=1000, max_angle=80.,
                 odf_state=True, odf_mode="CSD"):
        print("Loading dataset # ", dataset)
        self.device = device
        preprocessor = DataPreprocessor().normalize().crop(b_val).fa_estimate()
        if dataset == 'ISMRM':
            self.dataset = preprocessor.get_ismrm(f"data/ISMRM2015/")
        else:
            self.dataset = preprocessor.get_hcp(f"data/HCP/{dataset}/")

        self.stepWidth = stepWidth
        self.dtype = torch.FloatTensor  # vs. torch.cuda.FloatTensor
        self.dti_model = None
        self.dti_fit = None
        self.odf_interpolator = None
        self.shcoeff = None
        self.odf_mode = odf_mode

        np.random.seed(42)

        # phi = np.pi * np.random.rand(action_space)
        # theta = 2 * np.pi * np.random.rand(action_space)
        # sphere = HemiSphere(theta=theta, phi=phi)  #Sphere(theta=theta, phi=phi)
        # sphere, potential = disperse_charges(sphere, 5000) # enforce uniform distribtuion of our points
        # self.sphere = sphere
        self.sphere_odf = get_sphere('repulsion100')
        self.sphere = self.sphere_odf
        # print("sphere_odf = sphere_action = repulsion100")

        ## interpolation function of state's value ##
        self.state_interpol_fctn = self.interpolateRawDWIatState
        if (odf_state):
            print("Interpolating ODF as state Value")
            self.state_interpol_fctn = self.interpolateODFatState

        self.directions = torch.from_numpy(self.sphere.vertices).to(device)
        noActions, _ = self.directions.shape
        self.directions_odf = torch.from_numpy(self.sphere_odf.vertices).to(device)

        self.action_space = Discrete(noActions)  # spaces.Discrete(noActions+1)
        self.dwi_postprocessor = Resample(sphere=get_sphere('repulsion100'))  # resample(sphere=sphere)
        self.referenceStreamline_ijk = None
        self.grid = get_grid(np.array(grid_dim))
        self.maxL2dist_to_State = maxL2dist_to_State
        self.tracking_in_RAS = tracking_in_RAS

        ## load streamlines
        # self.changeReferenceStreamlinesFile(pReferenceStreamlines)
        self.fa_threshold = fa_threshold
        self.maxSteps = 2000

        ## init seeds
        self.seeds = seeds
        if self.seeds is None:
            if self.dti_fit is None:
                self._init_odf()

            gtab = gradient_table(self.dataset.bvals, self.dataset.bvecs)
            dti_model = dti.TensorModel(gtab, fit_method='LS')
            dti_fit = dti_model.fit(self.dataset.dwi, mask=self.dataset.binary_mask)

            fa_img = dti_fit.fa
            seed_mask = fa_img.copy()
            seed_mask[seed_mask >= 0.2] = 1
            seed_mask[seed_mask < 0.2] = 0

            seeds = utils.seeds_from_mask(seed_mask, affine=np.eye(4), density=1)  # tracking in IJK
            self.seeds = torch.from_numpy(seeds)

        self.reset()

        ## init adjacency matric
        self.max_angle = max_angle  # the maximum angle between two direction vectors
        self.cos_similarity = np.cos(
            np.deg2rad(max_angle))  # set cosine similarity treshold for initialization of adjacency matrix
        self._set_adjacency_matrix(self.sphere, self.cos_similarity)

        ## init obersavation space
        obs_shape = self.getObservationFromState(self.state).shape
        self.observation_space = Box(low=0, high=150, shape=obs_shape)

    def _set_adjacency_matrix(self, sphere, cos_similarity):
        """Creates a dictionary where each key is a direction from sphere and
        each value is a boolean array indicating which directions are less than
        max_angle degrees from the key"""
        matrix = np.dot(sphere.vertices, sphere.vertices.T)
        matrix = abs(matrix) >= cos_similarity
        keys = [tuple(v) for v in sphere.vertices]
        adj_matrix = dict(zip(keys, matrix))
        keys = [tuple(-v) for v in sphere.vertices]
        adj_matrix.update(zip(keys, matrix))
        self._adj_matrix = adj_matrix

    def _init_odf(self):
        print("Initialising ODF")
        gtab = gradient_table(self.dataset.bvals, self.dataset.bvecs)
        # fit DTI model to data
        if self.odf_mode == "DTI":
            print("DTI-based ODF computation")
            self.dti_model = dti.TensorModel(gtab, fit_method='LS')
            self.dti_fit = self.dti_model.fit(self.dataset.dwi, mask=self.dataset.binary_mask)
            # compute ODF
            odf = self.dti_fit.odf(self.sphere_odf)
        elif self.odf_mode == "CSD":
            print("CSD-based ODF computation")
            response, ratio = auto_response_ssst(gtab, self.dataset.dwi, roi_radii=10, fa_thr=0.7)
            mask = mask_for_response_ssst(gtab, self.dataset.dwi, roi_radii=10, fa_thr=0.7)
            nvoxels = np.sum(mask)
            print(nvoxels)
            response, ratio = response_from_mask_ssst(gtab, self.dataset.dwi, mask)
            print(response)
            self.dti_model = ConstrainedSphericalDeconvModel(gtab, response)
            self.dti_fit = self.dti_model.fit(self.dataset.dwi)
            odf = self.dti_fit.odf(self.sphere_odf)

        ## set up interpolator for odf evaluation
        x_range = np.arange(odf.shape[0])
        y_range = np.arange(odf.shape[1])
        z_range = np.arange(odf.shape[2])

        self.odf_interpolator = RegularGridInterpolator((x_range, y_range, z_range), odf)

        # print("Computing pmf")
        # self.pmf = odf.clip(min=0)

    def _init_shmcoeff(self, sh_basis_type=None):
        print("Initialising spherical harmonics")
        gtab = gradient_table(self.dataset.bvals, self.dataset.bvecs)
        self.dti_model = dti.TensorModel(gtab, fit_method='LS')

        peaks = peaks_from_model(model=self.dti_model, data=self.dataset.dwi, sphere=self.sphere, \
                                 relative_peak_threshold=.2, min_separation_angle=25, mask=self.dataset.binary_mask,
                                 npeaks=2)

        self.shcoeff = peaks.shm_coeff
        sh_order = order_from_ncoef(self.shcoeff.shape[-1])
        try:
            basis = sph_harm_lookup[sh_basis_type]
        except KeyError:
            raise ValueError("%s is not a known basis type." % sh_basis_type)

        self._B, m, n = basis(sh_order, self.sphere.theta, self.sphere.phi)

    def interpolateRawDWIatState(self, stateCoordinates):
        # TODO: maybe stay in RAS all the time then no need to transfer to IJK
        ras_points = self.dataset.to_ras(stateCoordinates)  # Transform state to World RAS+ coordinate system

        ras_points = self.grid + ras_points

        try:
            interpolated_dwi = self.dataset.get_interpolated_dwi(ras_points, postprocessing=self.dwi_postprocessor)
        except:
            # print("Point outside of brain mask :(")
            return None
        interpolated_dwi = np.rollaxis(interpolated_dwi, 3)  # CxWxHxD
        # interpolated_dwi = self.dtype(interpolated_dwi).to(self.device)
        return interpolated_dwi

    def interpolateODFatState(self, stateCoordinates):
        if self.odf_interpolator is None:
            self._init_odf()

        ijk_pts = self.grid + stateCoordinates.cpu().detach().numpy()
        interpol_odf = self.odf_interpolator(ijk_pts)
        interpol_odf = np.rollaxis(interpol_odf, 3)
        return interpol_odf

    ''' 
    @deprecated 
    '''

    def interpolatePMFatState(self, stateCoordinates):
        if self.shcoeff is None:
            self._init_shmcoeff()

        coeff = trilinear_interpolate4d(self.shcoeff, stateCoordinates)
        pmf = np.dot(self._B, coeff)
        pmf.clip(0, out=pmf)
        return pmf

    def step(self, action, direction="forward"):
        self.stepCounter += 1

        ### Termination conditions ###
        # I. number of steps larger than maximum
        if self.stepCounter == self.maxSteps:
            return self.state, 0., True, {}

            # II. fa below threshold? stop tracking
        if self.dataset.get_fa(self.state.getCoordinate()) < self.fa_threshold:
            return self.state, 0., True, {}

            ### Tracking ###
        cur_tangent = self.directions[action].view(-1, 3)
        cur_position = self.state.getCoordinate().view(-1, 3)
        if self.stepCounter == 1. and direction == "backward":
            cur_tangent = cur_tangent * -1
        next_position = cur_position + self.stepWidth * cur_tangent
        nextState = TractographyState(next_position, self.state_interpol_fctn)

        ### REWARD ###
        # compute reward based on
        # I. cosine similarity to peak direction of ODF (=> imitate maximum direction getter)
        # odf_peak_dir = self._get_best_action_ODF(cur_position).view(-1,3)
        # reward = abs(torch.nn.functional.cosine_similarity(odf_peak_dir, cur_tangent))

        # Inew. We basically take the normalized odf value corresponding to the encoded (action) tangent as reward
        #       It is normalized in a way such that its maximum equals 1
        #       Crucial assumption is that self.directions == self.directions_odf
        # @TODO: no. of diffusion directions hard-coded to 100
        # @TODO: taken center slice as this resembles maximum DG more closely. Alternatively: odf should be averaged first
        odf_cur = torch.from_numpy(self.interpolateODFatState(stateCoordinates=cur_position))[:, 1, 1, 1].view(100)
        reward = odf_cur / torch.max(odf_cur)
        reward = reward[action]

        # II. cosine similarity of current tangent to previous tangent 
        #     => Agent should prefer going straight
        if self.stepCounter > 1:
            prev_tangent = self.state_history[-1].getCoordinate() - self.state_history[-2].getCoordinate()
            prev_tangent = prev_tangent.view(-1, 3)
            prev_tangent = prev_tangent / torch.sqrt(torch.sum(prev_tangent ** 2, dim=1))  ## normalize to unit vector
            cos_similarity = torch.nn.functional.cosine_similarity(prev_tangent, cur_tangent)
            reward = (reward * cos_similarity).squeeze()
            if cos_similarity <= 0.:
                return nextState, reward, True, {}

        ### book keeping
        self.state_history.append(nextState)
        self.state = nextState

        return nextState, reward, False, {}

    def rewardForState(self, state, current_direction):
        my_position = state.getCoordinate().double().squeeze(0)
        ## main peak from ODF
        pmf_cur = self.interpolateODFatState(my_position)[:, 1, 1, 1]
        reward = pmf_cur / np.max(pmf_cur)
        if current_direction is not None:
            # reward = reward * self._adj_matrix[tuple(current_direction)] #@TODO: buggy
            reward = reward * (torch.nn.functional.cosine_similarity(self.directions,
                                                                     torch.from_numpy(current_direction).view(1,
                                                                                                              -1)).view(
                1, -1)).cpu().numpy()
        return reward

    def rewardForStateActionPair(self, state, current_direction, action):
        reward = self.rewardForState(state, current_direction)
        return reward[action]

    def _get_best_action(self, state, current_direction):
        reward = self.rewardForState(state, current_direction)
        best_action = np.argmax(reward)
        return best_action

    def track(self, withBestAction=True):
        streamlines = []
        for i in trange(len(self.seeds)):
            terminal = False
            all_states = []
            state = self.reset(seed_index=i)
            seed_position = state.getCoordinate().squeeze(0).numpy()
            current_direction = None
            all_states.append(seed_position)

            ## forward tracking
            terminal = False
            eval_steps = 0
            while not terminal:
                # current position
                # get best choice from environment
                if (withBestAction):
                    action = self._get_best_action(state, current_direction)
                # store tangent for next time step
                current_direction = self.directions[action].numpy()
                # take a step
                state, reward, terminal, _ = self.step(action)
                if (not terminal):
                    all_states.append(state.getCoordinate().squeeze(0).numpy())
                eval_steps = eval_steps + 1

            ## backward tracking
            state = self.reset(seed_index=i, terminal_F=True)
            current_direction = None  # potentially take tangent of first step of forward tracker
            terminal = False
            all_states = all_states[::-1]
            while not terminal:
                # current position
                my_position = state.getCoordinate().double().squeeze(0)
                # get best choice from environment
                if (withBestAction):
                    action = self._get_best_action(state, current_direction)
                # store tangent for next time step
                current_direction = self.directions[action].numpy()
                # take a step
                state, reward, terminal, _ = self.step(action, direction="backward")
                if (False in torch.eq(state.getCoordinate().squeeze(0), my_position)) & (not terminal):
                    all_states.append(state.getCoordinate().squeeze(0).numpy())

            streamlines.append(np.asarray(all_states))

        return streamlines

    def _get_multi_best_action_ODF(self, my_position, K=3):
        my_odf = self.odf_interpolator(my_position).squeeze()
        k_largest = np.argpartition(my_odf.squeeze(), -K)[-K:]
        peak_dirs_torch = self.directions_odf[k_largest].view(K, 3)
        rewards = torch.stack(
            [abs(torch.nn.functional.cosine_similarity(peak_dirs_torch[k:k + 1, :], self.directions.view(-1, 3))) for k
             in range(K)])
        return rewards

    def cosineSimilarity(self, path_1, path_2):
        return torch.nn.functional.cosine_similarity(path_2, path_1, dim=0)

    def arccos(self, angle):
        return torch.arccos(angle)

    def getObservationFromState(self, state):
        dwi_values = state.getValue().flatten()
        past_coordinates = np.array(list(self.state_history)).flatten()
        return np.concatenate((dwi_values, past_coordinates))

    # switch reference streamline of environment
    ##@TODO: optimize runtime
    def switchStreamline(self, streamline):
        self.line = streamline
        self.referenceStreamline_ijk = self.dtype(np.array(streamline.coords[:])).to(self.device)
        self.step_angles = []

    # reset the game and returns the observed data from the last episode
    def reset(self, seed_index=None, terminal_F=False, terminal_B=False):
        # self.seed_index = seed_index
        if seed_index is not None:
            self.seed_index = seed_index
        elif not terminal_F and not terminal_B or terminal_F and terminal_B:
            self.seed_index = np.random.randint(len(self.seeds))

        if (self.tracking_in_RAS):
            referenceSeedPoint_ras = self.seeds[self.seed_index]
            referenceSeedPoint_ijk = self.dataset.to_ijk(
                referenceSeedPoint_ras)  # könnte evtl. mit den shapes nicht hinhauen
        else:
            referenceSeedPoint_ijk = self.seeds[self.seed_index]

        # self.switchStreamline(geom.LineString(referenceStreamline_ijk))

        self.done = False
        self.stepCounter = 0
        self.reward = 0
        self.past_reward = 0
        self.points_visited = 1  # position_index

        # tracking_start_index = start_index
        # if(start_middle_of_streamline):
        #    tracking_start_index = len(self.referenceStreamline_ijk) // 2

        self.referenceSeedPoint_ijk = referenceSeedPoint_ijk
        self.state = TractographyState(self.referenceSeedPoint_ijk, self.state_interpol_fctn)
        self.state_history = deque(maxlen=4)
        while len(self.state_history) != 4:
            self.state_history.append(self.state)

        return self.state

    def render(self, mode="human"):
        pass

    '''
    deprecated: these functions were implimented via Shapely which is not applicable to 3d data.. it is basically
    ignoring the 3rd dimension in any operation
    
    
    def cosineSimtoStreamline(self, state, nextState):
        #current_index = np.min([self.points_visited,len(self.referenceStreamline_ijk)-1])
        current_index = np.min([self.closestStreamlinePoint(self.state) + 1, len(self.referenceStreamline_ijk)-1])
        path_vector = (nextState.getCoordinate() - state.getCoordinate()).squeeze(0)
        reference_vector = self.referenceStreamline_ijk[current_index]-self.referenceStreamline_ijk[current_index-1]
        cosine_sim = torch.nn.functional.cosine_similarity(path_vector, reference_vector, dim=0)
        return cosine_sim
    
    
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
    
    
    def minDistToStreamline(self, streamline, state):
        dist_streamline = torch.mean( (torch.FloatTensor(streamline.coords[:]) - state.getCoordinate())**2, dim = 1 )
        return torch.min(dist_streamline)
    
    def _get_multi_best_action(self, current_direction, my_position, K = 3):
        # main peak from ODF
        reward = self._get_multi_best_action_ODF(my_position, K)

        if(current_direction is not None):
            reward = reward * (torch.nn.functional.cosine_similarity(self.directions, current_direction)).view(1,-1)

        reward = torch.max(reward, axis = 0).values
        best_action = torch.argmax(reward)
        print("Max reward: %.2f" % (torch.max(reward).cpu().detach().numpy()))
        return best_action
    
    
    def _correct_direction(self, action_vector):
        # handle keeping the agent to go in the direction we want
        if self.stepCounter <= 1:                                                               # if no past states
            last_direction = self.referenceStreamline_ijk[1] - self.referenceStreamline_ijk[0]  # get the direction in which the reference steamline is going
        else: # else get the direction the agent has been going so far
            last_direction = self.state_history[-1].getCoordinate() - self.state_history[-2].getCoordinate()
            last_direction = last_direction.squeeze(0)

        if np.dot(action_vector, last_direction) < 0: # if the agent chooses the complete opposite direction
            action_vector = -1 * action_vector  # force it to follow the rough direction of the streamline
        return action_vector
    '''

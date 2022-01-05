from collections import deque

import dipy.reconst.dti as dti
import gym
import numpy as np
import torch
from dipy.core.interpolation import trilinear_interpolate4d
from dipy.core.sphere import HemiSphere, Sphere
from dipy.core.sphere import disperse_charges
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.csdeconv import (mask_for_response_ssst,
                                   response_from_mask_ssst)
from dipy.reconst.shm import order_from_ncoef, sph_harm_lookup
from dipy.tracking import utils
from gym.spaces import Discrete, Box
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange

from dfibert.data import DataPreprocessor, PointOutsideOfDWIError
from dfibert.data.postprocessing import Resample
from dfibert.util import get_grid
from ._state import TractographyState


from .neuroanatomical_utils import FiberBundleDataset, interpolate3dAt





class RLTractEnvironment(gym.Env):
    def __init__(self, device, seeds=None, step_width=0.8, dataset='100307', grid_dim=(3, 3, 3),
                 max_l2_dist_to_state=0.1, tracking_in_RAS=True, fa_threshold=0.1, b_val=1000, max_angle=80.,
                 odf_state=True, odf_mode="CSD", action_space=100):
        self.state_history = None
        self.reference_seed_point_ijk = None
        self.points_visited = None
        self.past_reward = None
        self.reward = None
        self.stepCounter = None
        self.done = None
        self.seed_index = None
        self.step_angles = None
        self.line = None
        print("Loading dataset # ", dataset)
        self.device = device
        preprocessor = DataPreprocessor().normalize().crop(b_val).fa_estimate()
        if dataset == 'ISMRM':
            self.dataset = preprocessor.get_ismrm(f"data/ISMRM2015/")
        else:
            self.dataset = preprocessor.get_hcp(f"data/HCP/{dataset}/")

        self.step_width = step_width
        self.dtype = torch.FloatTensor  # vs. torch.cuda.FloatTensor
        self.dti_model = None
        self.dti_fit = None
        self.odf_interpolator = None
        self.sh_coefficient = None
        self.odf_mode = odf_mode

        np.random.seed(42)
        action_space = action_space 
        phi = np.pi * np.random.rand(action_space)
        theta = 2 * np.pi * np.random.rand(action_space)
        sphere = HemiSphere(theta=theta, phi=phi)  #Sphere(theta=theta, phi=phi)
        sphere, _ = disperse_charges(sphere, 5000) # enforce uniform distribtuion of our points
        self.sphere = sphere
        self.sphere_odf = self.sphere

        # -- interpolation function of state's value --
        self.state_interpol_func = self.interpolate_dwi_at_state
        if odf_state:
            print("Interpolating ODF as state Value")
            self.state_interpol_func = self.interpolate_odf_at_state

        self.directions = torch.from_numpy(self.sphere.vertices).to(device)
        no_actions, _ = self.directions.shape
        self.directions_odf = torch.from_numpy(self.sphere_odf.vertices).to(device)

        self.action_space = Discrete(no_actions)  # spaces.Discrete(no_actions+1)
        self.dwi_postprocessor = Resample(sphere=get_sphere('repulsion100'))  # resample(sphere=sphere)
        self.referenceStreamline_ijk = None
        self.grid = get_grid(np.array(grid_dim))
        self.maxL2dist_to_State = max_l2_dist_to_state
        self.tracking_in_RAS = tracking_in_RAS

        # -- load streamlines --
        # self.changeReferenceStreamlinesFile(pReferenceStreamlines)
        self.fa_threshold = fa_threshold
        self.maxSteps = 2000

        # -- init seeds --
        self.seeds = seeds
        if self.seeds is None:
            if self.dti_fit is None:
                self._init_odf()

            dti_model = dti.TensorModel(self.dataset.gtab, fit_method='LS')
            dti_fit = dti_model.fit(self.dataset.dwi, mask=self.dataset.binary_mask)

            fa_img = dti_fit.fa
            seed_mask = fa_img.copy()
            seed_mask[seed_mask >= 0.2] = 1
            seed_mask[seed_mask < 0.2] = 0

            seeds = utils.seeds_from_mask(seed_mask, affine=np.eye(4), density=1)  # tracking in IJK
            self.seeds = torch.from_numpy(seeds)

        self.reset()

        # -- init adjacency matrix --
        self.max_angle = max_angle  # the maximum angle between two direction vectors
        self.cos_similarity = np.cos(
            np.deg2rad(max_angle))  # set cosine similarity threshold for initialization of adjacency matrix
        self._set_adjacency_matrix(self.sphere, self.cos_similarity)

        # -- init observation space --
        #obs_shape = self.get_observation_from_state(self.state).shape  ### <- TODO comment back in after debugging env
        #self.observation_space = Box(low=0, high=150, shape=obs_shape)

        # self.state = None  <-- is defined in reset function
        # -- init bundles for neuroanatomical loss --
        self.fibers_CST_left = FiberBundleDataset(path_to_files="data/ISMRM2015/gt_bundles/CST_left.fib")
        self.fibers_CST_right = FiberBundleDataset(path_to_files="data/ISMRM2015/gt_bundles/CST_right.fib")
        self.fibers_Fornix = FiberBundleDataset(path_to_files="data/ISMRM2015/gt_bundles/Fornix.fib")


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
        # fit DTI model to data
        if self.odf_mode == "DTI":
            print("DTI-based ODF computation")
            self.dti_model = dti.TensorModel(self.dataset.gtab, fit_method='LS')
            self.dti_fit = self.dti_model.fit(self.dataset.dwi, mask=self.dataset.binary_mask)
            # compute ODF
            odf = self.dti_fit.odf(self.sphere_odf)
        elif self.odf_mode == "CSD":
            print("CSD-based ODF computation")
            mask = mask_for_response_ssst(self.dataset.gtab, self.dataset.dwi, roi_radii=10, fa_thr=0.7)
            num_voxels = np.sum(mask)
            print(num_voxels)
            response, ratio = response_from_mask_ssst(self.dataset.gtab, self.dataset.dwi, mask)
            print(response)
            self.dti_model = ConstrainedSphericalDeconvModel(self.dataset.gtab, response)
            self.dti_fit = self.dti_model.fit(self.dataset.dwi)
            odf = self.dti_fit.odf(self.sphere_odf)

        # -- set up interpolator for odf evaluation
        x_range = np.arange(odf.shape[0])
        y_range = np.arange(odf.shape[1])
        z_range = np.arange(odf.shape[2])

        self.odf_interpolator = RegularGridInterpolator((x_range, y_range, z_range), odf)

        # print("Computing pmf")
        # self.pmf = odf.clip(min=0)

    def _init_shm_coefficient(self, sh_basis_type=None):
        print("Initialising spherical harmonics")
        self.dti_model = dti.TensorModel(self.dataset.gtab, fit_method='LS')

        peaks = peaks_from_model(model=self.dti_model, data=self.dataset.dwi, sphere=self.sphere,
                                 relative_peak_threshold=.2, min_separation_angle=25, mask=self.dataset.binary_mask,
                                 npeaks=2)

        self.sh_coefficient = peaks.shm_coeff
        sh_order = order_from_ncoef(self.sh_coefficient.shape[-1])
        try:
            basis = sph_harm_lookup[sh_basis_type]
        except KeyError:
            raise ValueError("%s is not a known basis type." % sh_basis_type)

        self._B, m, n = basis(sh_order, self.sphere.theta, self.sphere.phi)

    def interpolate_dwi_at_state(self, stateCoordinates):
        # TODO: maybe stay in RAS all the time then no need to transfer to IJK
        ras_points = self.dataset.to_ras(stateCoordinates)  # Transform state to World RAS+ coordinate system

        ras_points = self.grid + ras_points

        try:
            interpolated_dwi = self.dataset.get_interpolated_dwi(ras_points, postprocessing=self.dwi_postprocessor)
        except PointOutsideOfDWIError:
            # print("Point outside of brain mask :(")
            return None
        interpolated_dwi = np.rollaxis(interpolated_dwi, 3)  # CxWxHxD
        # interpolated_dwi = self.dtype(interpolated_dwi).to(self.device)
        return interpolated_dwi


    def interpolate_odf_at_state(self, stateCoordinates):
        if self.odf_interpolator is None:
            self._init_odf()

        ijk_pts = self.grid + stateCoordinates.cpu().detach().numpy()
        interpol_odf = self.odf_interpolator(ijk_pts)
        interpol_odf = np.rollaxis(interpol_odf, 3)
        return interpol_odf

    ''' 
    @deprecated 
    '''

    def interpolate_pmf_at_state(self, stateCoordinates):
        if self.sh_coefficient is None:
            self._init_shm_coefficient()

        coefficient = trilinear_interpolate4d(self.sh_coefficient, stateCoordinates)
        pmf = np.dot(self._B, coefficient)
        pmf.clip(0, out=pmf)
        return pmf

    def step(self, action, direction="forward"):
        self.stepCounter += 1

        # -- Termination conditions --
        # I. number of steps larger than maximum
        if self.stepCounter >= self.maxSteps:
            return self.get_observation_from_state(self.state), 0., True, {}

        # II. fa below threshold? stop tracking
        if self.dataset.get_fa(self.state.getCoordinate()) < self.fa_threshold:
            return self.get_observation_from_state(self.state), 0., True, {}
        
        #@todo: III. leaving brain mask
        #if self.dataset.get_fa(self.state.getCoordinate()) < self.fa_threshold:
        #    return self.get_observation_from_state(self.state), 0., True, {}

        # -- Tracking --
        cur_tangent = self.directions[action].view(-1, 3) # action space = Hemisphere
        cur_position = self.state.getCoordinate().view(-1, 3)
        if(direction == "backward"):
            cur_tangent = cur_tangent * -1
        next_position = cur_position + self.step_width * cur_tangent
        next_state = TractographyState(next_position, self.state_interpol_func)

        # -- REWARD --
        # compute reward based on
        # I. We basically take the normalized odf value corresponding to the encoded (action) tangent as reward
        # It is normalized in a way such that its maximum equals 1 Crucial assumption is that self.directions ==
        # self.directions_odf
        odf_cur = torch.from_numpy(self.interpolate_odf_at_state(stateCoordinates=cur_position))[:, 1, 1, 1].view(self.directions_odf.shape[0])
        reward = odf_cur / torch.max(odf_cur)
        reward = reward[action]

        # II. cosine similarity of current tangent to previous tangent 
        #     => Agent should prefer going straight
        prev_tangent = None
        if self.stepCounter > 1:
            prev_tangent = self.state_history[-1].getCoordinate() - self.state_history[-2].getCoordinate()
            prev_tangent = prev_tangent.view(-1, 3)
            prev_tangent = prev_tangent / torch.sqrt(torch.sum(prev_tangent ** 2, dim=1))  # normalize to unit vector
            cos_similarity = abs(torch.nn.functional.cosine_similarity(prev_tangent, cur_tangent))
            reward = (reward * cos_similarity).squeeze()
            if cos_similarity <= 0.:
                return next_state, reward, True, {}

        #@todo: replace the following lines by a call to reward_for_state_action_pair(-)
        #reward_sap = self.reward_for_state_action_pair(self.state, prev_tangent, action)
            
        # -- book keeping --
        self.state_history.append(next_state)
        self.state = next_state

        return self.get_observation_from_state(next_state), reward, False, {}

    def reward_for_state(self, state, prev_direction):
        my_position = state.getCoordinate().double().squeeze(0)
        # -- main peak from ODF --
        pmf_cur = self.interpolate_odf_at_state(my_position)[:, 1, 1, 1]
        reward = pmf_cur / np.max(pmf_cur)
        peak_indices = self._get_odf_peaks(reward, window_width=int(self.action_space.n/3))
        mask = np.zeros_like(reward)
        mask[peak_indices] = 1
        reward *= mask
        if current_direction is not None:
            reward = reward * abs(torch.nn.functional.cosine_similarity(self.directions, torch.from_numpy(prev_direction).view(1,-1))).view(1,-1).cpu().numpy()
        
        # neuroanatomical reward
        next_pos = my_position.view(1,-1) + self.directions
        reward_na = interpolate3dAt(self.fibers_Fornix.tractMask, next_pos) + interpolate3dAt(self.fibers_CST_left.tractMask, next_pos) + interpolate3dAt(self.fibers_CST_right.tractMask, next_pos)
        reward_na = reward_na.view(1, -1).cpu().numpy()
        return reward + reward_na

    def reward_for_state_action_pair(self, state, prev_direction, action):
        reward = self.reward_for_state(state, prev_direction)
        return reward[action]

    def _get_best_action(self, state, prev_direction):
        reward = self.reward_for_state(state, prev_direction)
        return np.argmax(reward)#best_action_d

    def _get_odf_peaks(self, odf, window_width=31):
        odf = torch.from_numpy(odf).squeeze(0)
        odf_diff = ((odf[:-2]<odf[1:-1]) & (odf[2:]<odf[1:-1])).type(torch.uint8)
        if window_width % 2 == 0.:
            window_width += 1
        peak_mask = torch.cat([torch.zeros(1, dtype=torch.uint8), odf_diff, torch.zeros(1, dtype=torch.uint8)], dim=0)
        peaks = torch.nn.functional.max_pool1d_with_indices(odf.view(1,1,-1), window_width, 1, padding=window_width//2)[1].unique()
        peak_indices = peaks[peak_mask[peaks].nonzero()]
        return peak_indices

    def track(self, with_best_action=True):
        streamlines = []
        for i in trange(len(self.seeds)):
            all_states = []
            self.reset(seed_index=i)
            state = self.state  # reset function now returns dwi values --> due to compatibility to rainbow agent or stable baselines
            seed_position = state.getCoordinate().squeeze(0).numpy()
            current_direction = None
            all_states.append(seed_position)

            # -- forward tracking
            terminal = False
            eval_steps = 0
            while not terminal:
                # current position
                # get the best choice from environment
                if with_best_action:
                    action = self._get_best_action(state, current_direction)
                else:
                    raise NotImplementedError
                # store tangent for next time step
                current_direction = self.directions[action].numpy()
                # take a step
                _, reward, terminal, _ = self.step(action)
                state = self.state # step function now returns dwi values --> due to compatibility to rainbow agent or stable baselines
                if not terminal:
                    all_states.append(state.getCoordinate().squeeze(0).numpy())
                eval_steps = eval_steps + 1

            # -- backward tracking
            self.reset(seed_index=i, terminal_F=True)
            state = self.state # reset function now returns dwi values --> due to compatibility to rainbow agent or stable baselines
            current_direction = None  # potentially take tangent of first step of forward tracker
            terminal = False
            all_states = all_states[::-1]
            while not terminal:
                # current position
                my_position = state.getCoordinate().double().squeeze(0)
                # get the best choice from environment
                if with_best_action:
                    action = self._get_best_action(state, current_direction)
                # store tangent for next time step
                current_direction = self.directions[action].numpy()
                # take a step
                _, reward, terminal, _ = self.step(action, direction="backward")
                state = self.state
                if (False in torch.eq(state.getCoordinate().squeeze(0), my_position)) & (not terminal):
                    all_states.append(state.getCoordinate().squeeze(0).numpy())

            streamlines.append(np.asarray(all_states))

        return streamlines

    def _get_multi_best_action_odf(self, my_position, K=3):
        my_odf = self.odf_interpolator(my_position).squeeze()
        k_largest = np.argpartition(my_odf.squeeze(), -K)[-K:]
        peak_dirs_torch = self.directions_odf[k_largest].view(K, 3)
        rewards = torch.stack(
            [abs(torch.nn.functional.cosine_similarity(peak_dirs_torch[k:k + 1, :], self.directions.view(-1, 3))) for k
             in range(K)])
        return rewards

    def get_observation_from_state(self, state):
        dwi_values = state#.getValue().flatten()
        # TODO -> currently only on dwi values, not on past states
        #past_coordinates = np.array(list(self.state_history)).flatten()
        #return np.concatenate((dwi_values, past_coordinates))
        return dwi_values

    # switch reference streamline of environment
    # @TODO: optimize runtime
    def switch_streamline(self, streamline):
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

        if self.tracking_in_RAS:
            reference_seed_point_ras = self.seeds[self.seed_index]
            reference_seed_point_ijk = self.dataset.to_ijk(
                reference_seed_point_ras)
        else:
            reference_seed_point_ijk = self.seeds[self.seed_index]

        # self.switch_streamline(geom.LineString(referenceStreamline_ijk))

        self.done = False
        self.stepCounter = 0
        self.reward = 0
        self.past_reward = 0
        self.points_visited = 1  # position_index

        # tracking_start_index = start_index
        # if(start_middle_of_streamline):
        #    tracking_start_index = len(self.referenceStreamline_ijk) // 2

        self.reference_seed_point_ijk = reference_seed_point_ijk
        self.state = TractographyState(self.reference_seed_point_ijk, self.state_interpol_func)
        self.state_history = deque([self.state]*4, maxlen=4)

        return self.get_observation_from_state(self.state)

    def render(self, mode="human"):
        pass
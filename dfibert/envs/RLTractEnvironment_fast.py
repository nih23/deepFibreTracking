from collections import deque
import os
import gym
import numpy as np
import torch
import dipy.reconst.dti as dti
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
from dfibert.data.postprocessing import Resample, Resample100
from dfibert.util import get_grid
from ._state import TractographyState


from .neuroanatomical_utils import FiberBundleDataset, interpolate3dAt
from .NARLTractEnvironment import TorchGridInterpolator


class RLTractEnvironment(gym.Env):
    def __init__(self, device, seeds=None, step_width=0.8, dataset='100307', grid_dim=(3, 3, 3),
                 max_l2_dist_to_state=0.1, tracking_in_RAS=False, fa_threshold=0.1, b_val=1000, 
                 odf_state=True, odf_mode="CSD", action_space=100, pFolderBundles = "data/gt_bundles/"):
        print("Will be deprecated by NARLTractEnvironment as soon as Jos fixes all bugs in the reward function.")
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
        self.na_reward_history = None
        self.av_na_reward = None
        self.past_bundle = None
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

        # build DWI object by interpolating at all IJK coordinates
        interpol_pts = None
        # permute into CxHxWxD
        self.dwi = torch.from_numpy(Resample100().process(self.dataset, None, self.dataset.dwi)).to(device=device).float()

        np.random.seed(42)
        action_space = action_space 
        phi = np.pi * np.random.rand(action_space)
        theta = 2 * np.pi * np.random.rand(action_space)
        sphere = HemiSphere(theta=theta, phi=phi)  #Sphere(theta=theta, phi=phi)
        sphere, _ = disperse_charges(sphere, 5000) # enforce uniform distribtuion of our points
        self.sphere = sphere
        self.sphere_odf = sphere

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
        self.grid = torch.from_numpy(self.grid).to(self.device)
        self.maxL2dist_to_State = max_l2_dist_to_state
        self.tracking_in_RAS = tracking_in_RAS

        # -- load streamlines --
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

        # -- init bundles for neuroanatomical reward --
        print("Init tract masks for neuroanatomical reward")
        fibers = []
        self.bundleNames = os.listdir(pFolderBundles)
        for fibFile in self.bundleNames:
            pFibre = pFolderBundles + fibFile
            #print(" @ " + pFibre)
            fibers.append(FiberBundleDataset(path_to_files=pFibre, dataset = self.dataset).tractMask)

        ## Define our interpolators
        self.tractMasks = torch.stack(fibers, dim = 0).to(self.device).permute((1,2,3,0)) # [X,Y,Z,C]
        print(self.tractMasks.shape)
        self.tractMask_interpolator = TorchGridInterpolator(self.tractMasks)
        self.binary_mask = torch.from_numpy(self.dataset.binary_mask).to(device=device)
        self.fa_interpolator = TorchGridInterpolator(torch.from_numpy(self.dataset.fa).to(device=device).unsqueeze(-1).float())
        self.dwi_interpolator = TorchGridInterpolator(self.dwi.to(self.device))
        self.brainMask_interpolator = TorchGridInterpolator(torch.from_numpy(self.dataset.binary_mask).to(self.device).unsqueeze(-1).float())

        # -- set default values --
        self.reset()


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
        odf = torch.from_numpy(odf).to(device=self.device).float()
        self.odf_interpolator = TorchGridInterpolator(odf)
        print("..done!")

    def interpolate_dwi_at_state(self, stateCoordinates):
        # torch
        ijk_pts = self.grid + stateCoordinates
        new_shape = (*ijk_pts.shape[:-1], -1)

        interpolated_dwi = self.dwi_interpolator(ijk_pts)
        interpolated_dwi = interpolated_dwi.reshape(new_shape)

        return interpolated_dwi


    def interpolate_odf_at_state(self, stateCoordinates):
        # torch
        if self.odf_interpolator is None:
            self._init_odf()

        new_shape = (*stateCoordinates.shape[:-1], -1)

        interpol_odf = self.odf_interpolator(stateCoordinates)
        interpol_odf = interpol_odf.reshape(new_shape)
        return interpol_odf


    def step(self, action, direction="forward"):
        self.stepCounter += 1
        cur_position = self.state.getCoordinate().view(-1, 3).to(self.device)


        # -- Termination conditions --
        # I. number of steps larger than maximum
        if self.stepCounter >= self.maxSteps:
            return self.get_observation_from_state(self.state), 0., True, {}

        # II. fa below threshold? stop tracking
        if(self.fa_interpolator(cur_position) < self.fa_threshold):
        #if self.dataset.get_fa(self.state.getCoordinate().cpu()) < self.fa_threshold:
            return self.get_observation_from_state(self.state), 0., True, {}
        
        # III. leaving brain mask
        if(self.brainMask_interpolator(cur_position) == 0):
            return self.get_observation_from_state(self.state), 0., True, {}

        # -- Tracking --
        cur_tangent = self.directions[action].view(-1, 3) # get direction from action (action = vertex id on (half)sphere)
        if(direction == "backward"):
            cur_tangent = cur_tangent * -1
        next_position = cur_position + self.step_width * cur_tangent
        next_state = TractographyState(next_position, self.state_interpol_func)

        # -- REWARD --
        '''
        prev_tangent = None
        if self.stepCounter > 1:
            # The following ops are done in CPU. The performance shouldnt suffer from that and we save 1 host-device copy.
            prev_tangent = self.state_history[-1].getCoordinate() - self.state_history[-2].getCoordinate()
            prev_tangent = prev_tangent.view(-1, 3)
            prev_tangent = prev_tangent / torch.sqrt(torch.sum(prev_tangent ** 2, dim=1))  # normalize to unit vector
            prev_tangent = prev_tangent.to(self.device)
        '''
        
        reward = self.reward_for_state_action_pair(self.state, action, direction) # prev_tangent => None

        # -- book keeping --
        self.state_history.append(next_state)
        self.state = next_state

        return self.get_observation_from_state(next_state), reward, False, {}


    def reward_for_state(self, state, direction, prev_direction = None):
        my_position = state.getCoordinate().squeeze(0).to(self.device)
        # -- main peaks from ODF --
        pmf_cur = self.interpolate_odf_at_state(my_position)
        reward = pmf_cur / torch.max(pmf_cur)
        
        #if(prev_direction != None):
        #    print("[Warning] cosine similarity loss not used anymore due to resutls of ablation study.")
        
        ## ablation study on CST found that peak finding not required
        '''
        peak_indices = self._get_odf_peaks(reward, window_width=int(self.action_space.n/3))
        mask = torch.zeros_like(reward, device = self.device)
        mask[peak_indices] = 1
        reward *= mask
        '''
        
        ## ablation study on CST found that angular deviation not needed
        '''
        # -- limit angular deviation --
        if prev_direction is not None:
            reward = reward * abs(torch.nn.functional.cosine_similarity(self.directions, prev_direction.view(1,-1))).view(-1) # noActions
        # neuroanatomical reward
        '''
        
        
        # -- neuroanatomical reward --
        orientation = self.step_width * self.directions
        if(direction == "backward"):
            orientation = -1 * orientation
        next_pos = my_position.view(1,-1) + orientation # gets next positions for all directions actions X 3
        local_reward_na = self.tractMask_interpolator(next_pos) # noActions x noTracts
       
        reward_na_mu_hist = torch.mean(self.na_reward_history[0:max(self.stepCounter-1,1), :], dim = 0).view(1,-1) # 1 x no_tracts
        local_reward_na = local_reward_na + reward_na_mu_hist # noActions x noTracts
        reward_na, _ = torch.max(local_reward_na, dim = 1) # # marginalize tracts
        reward_na = reward_na.view(-1) # noActions 

        # reward_na_arg = torch.argmax(local_reward_na, dim = 0) # get dominant tract per action        

        return reward + reward_na # ODF + neuroanatomical reward

    
    def reward_for_state_action_pair(self, state, action, direction, prev_direction = None):
        reward = self.reward_for_state(state, direction, prev_direction)
        return reward[action]

    
    def _get_best_action(self, state, direction="forward", prev_direction = None):
        reward = self.reward_for_state(state, direction, prev_direction)
        return torch.argmax(reward)

    '''
    #@TODO: improve documentation
    #deprecated
    def _get_odf_peaks(self, odf, window_width=31):
        odf = odf.squeeze(0)
        odf_diff = ((odf[:-2]<odf[1:-1]) & (odf[2:]<odf[1:-1])).type(torch.uint8) #.to(self.device)
        if window_width % 2 == 0.:
            window_width += 1
        peak_mask = torch.cat([torch.zeros(1, dtype=torch.uint8, device = self.device), odf_diff, torch.zeros(1, dtype=torch.uint8, device = self.device)], dim=0)
        peaks = torch.nn.functional.max_pool1d_with_indices(odf.view(1,1,-1), window_width, 1, padding=window_width//2)[1].unique()
        peak_indices = peaks[peak_mask[peaks].nonzero()]
        return peak_indices
    '''

    
    def track(self, with_best_action=True):
        streamlines = []
        for i in trange(len(self.seeds)):
            all_states = []
            self.reset(seed_index=i)
            state = self.state  # reset function now returns dwi values --> due to compatibility to rainbow agent or stable baselines
            seed_position = state.getCoordinate().to(self.device)
            current_direction = None
            all_states.append(seed_position.squeeze(0))

            # -- forward tracking --
            terminal = False
            eval_steps = 0
            while not terminal:
                # current position
                # get the best choice from environment
                if with_best_action:
                    action = self._get_best_action(state, direction="forward", prev_direction=current_direction)
                else:
                    raise NotImplementedError
                # store tangent for next time step
                current_direction = self.directions[action] #.numpy()
                # take a step
                _, reward, terminal, _ = self.step(action)
                state = self.state # step function now returns dwi values --> due to compatibility to rainbow agent or stable baselines
                if not terminal:
                    all_states.append(state.getCoordinate().squeeze(0))
                eval_steps = eval_steps + 1

            # -- backward tracking --
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
                    action = self._get_best_action(state, direction="backward", prev_direction=current_direction)
                else:
                    raise NotImplementedError
                # store tangent for next time step
                current_direction = self.directions[action]#.numpy()
                # take a step
                _, reward, terminal, _ = self.step(action, direction="backward")
                state = self.state
                my_position = my_position.to(self.device) # DIRTY!!!
                my_coord = state.getCoordinate().squeeze(0).to(self.device)
                if (False in torch.eq(my_coord, my_position)) & (not terminal):
                    all_states.append(my_coord)

            streamlines.append((all_states))

        return streamlines


    def get_observation_from_state(self, state):
        dwi_values = state.getValue().flatten()
        # TODO -> currently only on dwi values, not on past states
        #past_coordinates = np.array(list(self.state_history)).flatten()
        #return np.concatenate((dwi_values, past_coordinates))
        return dwi_values


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

        self.done = False
        self.stepCounter = 0
        self.reward = 0
        self.past_reward = 0
        self.points_visited = 1  # position_index

        self.reference_seed_point_ijk = reference_seed_point_ijk.to(self.device)
        self.state = TractographyState(self.reference_seed_point_ijk, self.state_interpol_func)
        self.state_history = deque([self.state]*4, maxlen=4)
        self.na_reward_history = torch.zeros((self.maxSteps, self.tractMasks.shape[-1]), device = self.device) 

        return self.get_observation_from_state(self.state)

    def render(self, mode="human"):
        pass

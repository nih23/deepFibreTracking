import os
from pathlib import Path
from typing import Optional

import dipy.reconst.dti as dti
import gym
import numpy as np
from dipy.core.sphere import HemiSphere
from dipy.core.sphere import disperse_charges
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.csdeconv import (mask_for_response_ssst,
                                   response_from_mask_ssst)
from dipy.tracking import utils
from gym.spaces import Discrete
from tqdm import trange

from dfibert.data import DataPreprocessor
from dfibert.data.postprocessing import Resample, Resample100
from dfibert.util import get_grid
from ..tracker import load_streamlines

import torch

class TorchGridInterpolator:
    def __init__(self, data) -> None:
        self.data = data.float() # [X,Y,Z,C]
        
        self.data = self.data.moveaxis(3,0).unsqueeze(0) # [1, C, X, Y, Z]
        self.interpol_transform = (torch.tensor(data.shape[:3], device=self.data.device) - 1) / 2
    
    def _convert_points(self, pts):
        return (pts[:, (2,1,0)] / self.interpol_transform) - 1

    def __call__(self, pts) -> None:
        new_shape = (*pts.shape[:-1], -1)
        pts = pts.reshape(-1, 3)
        pts = self._convert_points(pts.float())
        pts = pts.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #print(pts.dtype, pts.shape)
        interpolated =  torch.nn.functional.grid_sample(self.data, pts, align_corners=True, mode="bilinear")
        # [1 , C, 1, 1, N]
        return interpolated.reshape((self.data.shape[1], -1)).moveaxis(1,0).reshape(new_shape)

def get_uniform_hemisphere_with_points(action_space: int, seed=42) -> HemiSphere:
    if seed is not None:
        np.random.seed(seed)

    phi = np.pi * np.random.rand(action_space)
    theta = 2 * np.pi * np.random.rand(action_space)
    sphere = HemiSphere(theta=theta, phi=phi)  # Sphere(theta=theta, phi=phi)
    sphere, _ = disperse_charges(sphere, 5000)  # enforce uniform distribution of our points

    return sphere


def get_tract_mask(path_to_files, dataset):
    streamlines = load_streamlines(path=str(path_to_files))
    streamlines = [dataset.to_ijk(sl) for sl in streamlines]
    tract_mask = np.zeros(dataset.binary_mask.shape)

    for sl in streamlines:
        pi = np.floor(sl).astype(int)
        tract_mask[pi[:, 0], pi[:, 1], pi[:, 2]] = 1
    return tract_mask


def get_all_tract_mask(bundle_path, dataset):
    tract_masks = np.stack([get_tract_mask(bundle_path / file, dataset=dataset) for file in os.listdir(bundle_path)])
    return np.moveaxis(tract_masks, 0, -1)  # X * Y * Z * Bundle


class TractographyState:
    def __init__(self, coordinate, interpol_func):
        self.coordinate = coordinate
        self.interpol_func = interpol_func
        self.interpol_dwi = None

    def get_coordinate(self):
        return self.coordinate

    def get_value(self):
        if self.interpol_dwi is None:
            # interpolate DWI value at self.coordinate
            self.interpol_dwi = self.interpol_func(self.coordinate)
        return self.interpol_dwi

    def __add__(self, other):
        if isinstance(other, torch.Tensor):
            return TractographyState(self.get_coordinate() + other, self.interpol_func)
        elif isinstance(other, TractographyState):
            return TractographyState(self.get_coordinate() + other.get_coordinate(), self.interpol_func)
        else:
            raise NotImplementedError()

    def __sub__(self, other):
        if isinstance(other, torch.Tensor):
            return TractographyState(self.get_coordinate() - other, self.interpol_func)
        elif isinstance(other, TractographyState):
            return TractographyState(self.get_coordinate() - other.get_coordinate(), self.interpol_func)
        else:
            raise NotImplementedError()


class NARLTractEnvironment(gym.Env):
    def __init__(self, device, seeds=None, dataset="100307", step_width=0.8, b_val=1000, action_space=100,
                 grid_dim=(3, 3, 3),
                 max_steps=2000, fa_threshold=0.2, bundles_path="data/gt_bundles/", odf_mode="CSD"):

        print("Loading dataset # ", dataset)
        self.device = device
        preprocessor = DataPreprocessor().normalize().crop(b_val).fa_estimate()
        if dataset == 'ISMRM':
            self.dataset = preprocessor.get_ismrm(f"data/ISMRM2015/")
        else:
            self.dataset = preprocessor.get_hcp(f"data/HCP/{dataset}/")
        self.sphere = get_uniform_hemisphere_with_points(action_space=action_space)
        self.directions = torch.from_numpy(self.sphere.vertices).to(device=device)
        self.grid = torch.from_numpy(get_grid(np.array(grid_dim))).to(device=device)
        self.action_space = Discrete(action_space)

        if seeds is None:
            seeds = utils.seeds_from_mask(self.dataset.binary_mask, self.dataset.aff)
        self.seeds = torch.from_numpy(seeds).to(device=device).float()  # RAS
        self.max_steps = max_steps
        self.step_width = step_width

        self.dwi = torch.from_numpy(Resample100().process(self.dataset, None, self.dataset.dwi)).to(device=device)
        self.dwi_processor = TorchGridInterpolator(self.dwi)
        self.binary_mask = torch.from_numpy(self.dataset.binary_mask).to(device=device)
        self.fa_interpolator = TorchGridInterpolator(torch.from_numpy(self.dataset.fa).to(device=device).unsqueeze(-1))
        self.fa_threshold = fa_threshold

        self._init_na(Path(bundles_path))
        self._init_odf(odf_mode=odf_mode)
        self.ras_aff = torch.from_numpy(self.dataset.aff).to(device=device).float()
        self.ijk_aff = self.ras_aff.inverse().float()
        self.state: Optional[TractographyState] = None
        self.no_steps = 0
        self.state_history = torch.zeros((self.max_steps + 1, 3))
        self.na_reward_history = torch.zeros((self.max_steps, self.tract_masks.shape[-1]))
        self.reset()

    def _init_na(self, bundles_path):
        self.tract_masks = torch.from_numpy(get_all_tract_mask(bundles_path, self.dataset)).to(device=self.device)
        self.na_interpolator = TorchGridInterpolator(self.tract_masks)

    def _init_odf(self, odf_mode):
        print("Initialising ODF")
        # fit DTI model to data
        if odf_mode == "DTI":
            print("DTI-based ODF computation")
            dti_model = dti.TensorModel(self.dataset.gtab, fit_method='LS')
            dti_fit = dti_model.fit(self.dataset.dwi, mask=self.dataset.binary_mask)
            # compute ODF
            odf = dti_fit.odf(self.sphere)
        elif odf_mode == "CSD":
            print("CSD-based ODF computation")
            mask = mask_for_response_ssst(self.dataset.gtab, self.dataset.dwi, roi_radii=10, fa_thr=0.7)
            response, ratio = response_from_mask_ssst(self.dataset.gtab, self.dataset.dwi, mask)
            dti_model = ConstrainedSphericalDeconvModel(self.dataset.gtab, response)
            dti_fit = dti_model.fit(self.dataset.dwi)
            odf = dti_fit.odf(self.sphere)
        else:
            raise NotImplementedError("ODF Mode not found")
        # -- set up interpolator for odf evaluation
        odf = torch.from_numpy(odf).to(device=self.device)

        self.odf_interpolator = TorchGridInterpolator(odf)

    def step(self, action, backwards=False):
        ijk_coordinate = self.state.get_coordinate()
        odf_cur = self.odf_interpolator(ijk_coordinate).squeeze()

        if torch.max(odf_cur) > 0:
            odf_cur = odf_cur / torch.max(odf_cur)

        if self.no_steps >= self.max_steps:
            return self.state.get_value(), 0., True, {}
        if self.fa_interpolator(ijk_coordinate) < self.fa_threshold:
            return self.state.get_value(), 0., True, {}

        if self.binary_mask[int(ijk_coordinate[0]), int(ijk_coordinate[1]), int(ijk_coordinate[2])] == 0:
            return self.state.get_value(), 0., True, {}

        next_dir = self.directions[action].clone().detach().float()
        if self.no_steps > 0:
            prev_dir = self.state_history[self.no_steps] - self.state_history[self.no_steps - 1]
            prev_dir = prev_dir / torch.linalg.norm(prev_dir)
            if torch.dot(next_dir, prev_dir) < 0:
                next_dir = next_dir * -1

            if torch.dot(next_dir, prev_dir) < 0.5:
                return self.state.get_value(), 0., True, {}
        else:
            if backwards:
                next_dir = next_dir * -1
            prev_dir = next_dir

        step_width = self.step_width if self.no_steps > 0 else 0.5 * self.step_width
        self.state = self.state + (step_width * next_dir)
        self.no_steps += 1
        self.state_history[self.no_steps] = self.state.get_coordinate()

        ijk_coordinate = self.state.get_coordinate()

        local_na_reward = self.na_interpolator(ijk_coordinate)

        self.na_reward_history[self.no_steps - 1, :] = local_na_reward

        if self.no_steps > 1:
            mean_na_reward = torch.mean(self.na_reward_history[0: self.no_steps - 1], dim=0)
            na_reward = mean_na_reward + local_na_reward
        else:
            na_reward = local_na_reward
        reward = odf_cur[action] * torch.dot(next_dir, prev_dir) + torch.max(na_reward)
        return self.state.get_value(), reward, False, {}

    def to_ras(self, points):
        new_shape = (points.shape)
        points = points.reshape(-1, 3)
        return (torch.mm(self.ras_aff[:3,:3],points.T) + self.ras_aff[:3,3:4]).T.reshape(new_shape)

    def to_ijk(self, points):
        new_shape = (points.shape)
        points = points.reshape(-1, 3)
        return (torch.mm(self.ijk_aff[:3,:3],points.T) + self.ijk_aff[:3,3:4]).T.reshape(new_shape)
    
    
    def interpolate_dwi_at_state(self, points):
        ras_points = self.grid + self.to_ras(points.float())


        new_shape = (*points.shape[:-1], -1)

        points = self.to_ijk(ras_points.float()).reshape(-1, 3)

        is_outside = ((points[:, 0] < 0) + (points[:, 0] >= self.dwi.shape[0]) +  # OR
                      (points[:, 1] < 0) + (points[:, 1] >= self.dwi.shape[1]) +
                      (points[:, 2] < 0) + (points[:, 2] >= self.dwi.shape[2])) > 0

        if torch.sum(is_outside) > 0:
            return None

        result = self.dwi_processor(points)

        
        result = result.reshape(new_shape)
        return result

    def _next_pos_and_reward(self, backwards=False):
        next_dirs = self.directions.clone().detach()
        if self.no_steps > 0:
            prev_dir = self.state_history[self.no_steps] - self.state_history[self.no_steps - 1]
            prev_dir = prev_dir / torch.linalg.norm(prev_dir)

            should_be_inverted = torch.sum(next_dirs * prev_dir, dim=1) < 0
            next_dirs[should_be_inverted] = -next_dirs[should_be_inverted]
        elif backwards:
            next_dirs = -next_dirs
        rewards = self._get_reward_for_move(torch.arange(0, self.directions.shape[0]), next_dirs)
        if self.no_steps > 0:
            angle_to_sharp = torch.sum(next_dirs * prev_dir, dim=1) < 0.5
            rewards[angle_to_sharp] = 0
        return next_dirs, rewards

    def _get_reward_for_move(self, actions, next_directions):

        cur_pos_ijk = self.state.get_coordinate()

        odf_cur = self.odf_interpolator(cur_pos_ijk.unsqueeze(0)).squeeze()  # [L]
        if torch.max(odf_cur) > 0:
            odf_cur = odf_cur / torch.max(odf_cur)
        # actions : [N], next_directions: [N ,3]
        step_width = self.step_width if self.no_steps > 0 else 0.5 * self.step_width

        next_positions = cur_pos_ijk + next_directions * step_width
        ijk_coordinates = next_positions  # [N, 3]
        local_na_reward = self.na_interpolator(ijk_coordinates.unsqueeze(0))  # [N, K]

        if self.no_steps > 0:
            mean_na_reward = torch.mean(self.na_reward_history[0: self.no_steps], dim=0)  # [K]
            na_reward = mean_na_reward + local_na_reward  # [N, K]

            prev_dir = self.state_history[self.no_steps] - self.state_history[self.no_steps - 1]
            prev_dir = prev_dir / torch.linalg.norm(prev_dir)
            return odf_cur[actions] * torch.sum(next_directions* prev_dir, dim=1) + torch.max(na_reward)
        else:
            return odf_cur[actions] + torch.max(local_na_reward)

    def track(self, with_best_action=True):
        streamlines = []
        for i in trange(len(self.seeds)):
            streamline = []
            self.reset(seed_index=i)

            # -- forward tracking --
            terminal = False
            while not terminal:
                # current position
                # get the best choice from environment
                if with_best_action:
                    _, reward = self._next_pos_and_reward()
                    action = torch.argmax(reward)
                else:
                    raise NotImplementedError
                # take a step
                _, reward, terminal, _ = self.step(action)
                # step function now returns dwi values --> due to compatibility to rainbow agent or stable baselines
                if not terminal:
                    streamline.append(self.to_ras(self.state.get_coordinate().float()).cpu().detach().numpy())

            # -- backward tracking --
            self.reset(seed_index=i)
            # reset function now returns dwi values --> due to compatibility to rainbow agent or stable baselines

            streamline = streamline[::-1]

            while not terminal:
                if with_best_action:
                    _, reward = self._next_pos_and_reward(backwards=True)
                    action = torch.argmax(reward)
                else:
                    raise NotImplementedError
                # take a step
                _, reward, terminal, _ = self.step(action, backwards=True)
                # step function now returns dwi values --> due to compatibility to rainbow agent or stable baselines
                if not terminal:
                    streamline.append(self.to_ras(self.state.get_coordinate().float()).cpu().detach().numpy())

            streamlines.append(streamline)

        return streamlines

    # reset the game and returns the observed data from the last episode
    def reset(self, seed_index=None):
        if seed_index is None:
            seed_index = torch.randint(len(self.seeds), size=(1,1))[0][0]

        self.seed_index = seed_index
        self.no_steps = 0
        self.state_history[:] = 0
        self.na_reward_history[:] = 0
        self.state_history[0] = self.to_ijk(self.seeds[seed_index].float())
        self.state = TractographyState(self.state_history[0], self.interpolate_dwi_at_state)

        return self.state.get_value()

    def render(self, mode="human"):
        pass

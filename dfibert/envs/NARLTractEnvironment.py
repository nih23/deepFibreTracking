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
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange

from dfibert.data import DataPreprocessor, PointOutsideOfDWIError
from dfibert.data.postprocessing import Resample, Resample100
from dfibert.util import get_grid
from ..tracker import load_streamlines


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
        if isinstance(other, np.ndarray):
            return TractographyState(self.get_coordinate() + other, self.interpol_func)
        elif isinstance(other, TractographyState):
            return TractographyState(self.get_coordinate() + other.get_coordinate(), self.interpol_func)
        else:
            raise NotImplementedError()

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
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
        preprocessor = DataPreprocessor().normalize().crop(b_val).fa_estimate()
        if dataset == 'ISMRM':
            self.dataset = preprocessor.get_ismrm(f"data/ISMRM2015/")
        else:
            self.dataset = preprocessor.get_hcp(f"data/HCP/{dataset}/")
        self.sphere = get_uniform_hemisphere_with_points(action_space=action_space)
        self.directions = self.sphere.vertices
        self.grid = get_grid(np.array(grid_dim))
        self.action_space = Discrete(action_space)

        if seeds is None:
            seeds = utils.seeds_from_mask(self.dataset.binary_mask, self.dataset.aff)
        self.seeds = seeds  # RAS

        self.max_steps = max_steps
        self.step_width = step_width

        self.dwi_postprocessor = Resample100()
        self.fa_threshold = fa_threshold

        self._init_na(Path(bundles_path))
        self._init_odf(odf_mode=odf_mode)
        self.state: Optional[TractographyState] = None
        self.no_steps = 0
        self.state_history = np.zeros((self.max_steps + 1, 3))
        self.na_reward_history = np.zeros((self.max_steps, self.na_interpolator.values.shape[-1]))
        self.reset()

    def _init_na(self, bundles_path):
        tract_masks = get_all_tract_mask(bundles_path, self.dataset)
        x = np.arange(tract_masks.shape[0])
        y = np.arange(tract_masks.shape[1])
        z = np.arange(tract_masks.shape[2])
        self.na_interpolator = RegularGridInterpolator((x, y, z), tract_masks)

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
        x_range = np.arange(odf.shape[0])
        y_range = np.arange(odf.shape[1])
        z_range = np.arange(odf.shape[2])

        self.odf_interpolator = RegularGridInterpolator((x_range, y_range, z_range), odf)

    def step(self, action, backwards=False):
        ijk_coordinate = self.state.get_coordinate()
        odf_cur = self.odf_interpolator(ijk_coordinate).squeeze()

        if np.max(odf_cur) > 0:
            odf_cur = odf_cur / np.max(odf_cur)

        if self.no_steps >= self.max_steps:
            return self.state.get_value(), 0., True, {}
        if self.dataset.get_fa(ijk_coordinate) < self.fa_threshold:
            return self.state.get_value(), 0., True, {}

        if self.dataset.binary_mask[int(ijk_coordinate[0]), int(ijk_coordinate[1]), int(ijk_coordinate[2])] == 0:
            return self.state.get_value(), 0., True, {}

        next_dir = np.copy(self.directions[action])

        if self.no_steps > 0:
            prev_dir = self.state_history[self.no_steps] - self.state_history[self.no_steps - 1]
            prev_dir = prev_dir / np.linalg.norm(prev_dir)
            if np.dot(next_dir, prev_dir) < 0:
                next_dir = next_dir * -1

            if np.dot(next_dir, prev_dir) < 0.5:
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
            mean_na_reward = np.mean(self.na_reward_history[0: self.no_steps - 1], axis=0)
            na_reward = mean_na_reward + local_na_reward
        else:
            na_reward = local_na_reward
        reward = odf_cur[action] * np.dot(next_dir, prev_dir) + np.max(na_reward)
        return self.state.get_value(), reward, False, {}

    def interpolate_dwi_at_state(self, coords):
        ras_points = self.grid + self.dataset.to_ras(coords)
        try:
            return self.dataset.get_interpolated_dwi(ras_points, postprocessing=self.dwi_postprocessor).flatten()
        except PointOutsideOfDWIError:
            return None

    def _next_pos_and_reward(self, backwards=False):
        next_dirs = np.copy(self.directions)
        if self.no_steps > 0:
            prev_dir = self.state_history[self.no_steps] - self.state_history[self.no_steps - 1]
            prev_dir = prev_dir / np.linalg.norm(prev_dir)

            should_be_inverted = np.sum(next_dirs * prev_dir, axis=1) < 0
            next_dirs[should_be_inverted] = -next_dirs[should_be_inverted]
        elif backwards:
            next_dirs = -next_dirs
        rewards = self._get_reward_for_move(np.arange(0, self.directions.shape[0]), next_dirs)
        if self.no_steps > 0:
            angle_to_sharp = np.sum(next_dirs * prev_dir, axis=1) < 0.5
            rewards[angle_to_sharp] = 0
        return next_dirs, rewards

    def _get_reward_for_move(self, actions, next_directions):

        cur_pos_ijk = self.state.get_coordinate()

        odf_cur = self.odf_interpolator(cur_pos_ijk).squeeze()  # [L]
        if np.max(odf_cur) > 0:
            odf_cur = odf_cur / np.max(odf_cur)
        # actions : [N], next_directions: [N ,3]
        step_width = self.step_width if self.no_steps > 0 else 0.5 * self.step_width

        next_positions = cur_pos_ijk + next_directions * step_width
        ijk_coordinates = next_positions  # [N, 3]
        local_na_reward = self.na_interpolator(ijk_coordinates)  # [N, K]

        if self.no_steps > 0:
            mean_na_reward = np.mean(self.na_reward_history[0: self.no_steps], axis=0)  # [K]
            na_reward = mean_na_reward + local_na_reward  # [N, K]

            prev_dir = self.state_history[self.no_steps] - self.state_history[self.no_steps - 1]
            prev_dir = prev_dir / np.linalg.norm(prev_dir)
            return odf_cur[actions] * np.dot(next_directions, prev_dir) + np.max(na_reward)
        else:
            return odf_cur[actions] + np.max(local_na_reward)

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
                    action = np.argmax(reward)
                else:
                    raise NotImplementedError
                # take a step
                _, reward, terminal, _ = self.step(action)
                # step function now returns dwi values --> due to compatibility to rainbow agent or stable baselines
                if not terminal:
                    streamline.append(self.dataset.to_ras(self.state.get_coordinate()))

            # -- backward tracking --
            self.reset(seed_index=i)
            # reset function now returns dwi values --> due to compatibility to rainbow agent or stable baselines

            streamline = streamline[::-1]

            while not terminal:
                if with_best_action:
                    _, reward = self._next_pos_and_reward(backwards=True)
                    action = np.argmax(reward)
                else:
                    raise NotImplementedError
                # take a step
                _, reward, terminal, _ = self.step(action, backwards=True)
                # step function now returns dwi values --> due to compatibility to rainbow agent or stable baselines
                if not terminal:
                    streamline.append(self.dataset.to_ras(self.state.get_coordinate()))

            streamlines.append(streamline)

        return streamlines

    # reset the game and returns the observed data from the last episode
    def reset(self, seed_index=None):
        if seed_index is None:
            seed_index = np.random.randint(len(self.seeds))

        self.seed_index = seed_index
        self.no_steps = 0
        self.state_history[:] = 0
        self.na_reward_history[:] = 0

        self.state_history[0] = self.dataset.to_ijk(self.seeds[seed_index])
        self.state = TractographyState(self.state_history[0], self.interpolate_dwi_at_state)

        return self.state.get_value()

    def render(self, mode="human"):
        pass

"""
The data module is handling all kinds of DWI-data.

Use this as a starting point to represent your loaded DWI-scan.
This module provides methods helping you to implement datasets, 
environments and all other kinds of modules with the requirement
to work directly with the data.  
"""
from __future__ import annotations

import collections
import os
import warnings
from collections import Callable

import dipy.reconst.dti as dti
import nibabel as nb
import numpy
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.io import read_bvals_bvecs
from dipy.segment.mask import median_otsu
from nibabel.affines import apply_affine
from scipy.interpolate import RegularGridInterpolator

from dfibert.data.exceptions import (DeviceNotRetrievableError, DataContainerNotLoadableError, DWIAlreadyCroppedError,
                                     DWIAlreadyNormalizedError, PointOutsideOfDWIError)


class DataPreprocessor(object):
    def __init__(self, parent: DataPreprocessor = None):
        """
        Creates a new empty DataPreprocessor.

        :param parent: An optional previous DataPreprocessor we want to continue with
        :type parent: DataPreprocessor, optional
        """
        self._parent = parent

    def _preprocess(self, data_container: DataContainer) -> DataContainer:
        """
        :param data_container: a DataContainer
        :return: a tuple with the identical parameters processed
        """
        if self._parent is None:
            return data_container
        else:
            return self._parent._preprocess(data_container)

    def preprocess(self, data_container: DataContainer) -> DataContainer:
        """
        Returns a preprocessed DataContainer created by taking the given one and applying the given steps.
        Because data_containers are treated as immutable, the given data_container (and its numpy arrays)
        won't be modified.

        :param data_container: The given data_container
        :return: A new preprocessed data_container
        """
        bvals, bvecs, t1, dwi, aff, binary_mask, b0, fa = data_container
        data_container = DataContainer(bvals.copy(), bvecs.copy(), t1.copy(), dwi.copy(), aff.copy(),
                                       binary_mask.copy(), b0.copy(), None if fa is None else fa.copy())
        return self._preprocess(data_container)

    def denoise(self, smooth=3, patch_radius=3) -> DataPreprocessor:
        """Denoises the data using Local PCA with empirical thresholds

        :param smooth: the voxel radius used by the Gaussian filter for the noise estimate
        :param patch_radius: the voxel radius used by the Local PCA algorithm to denoise
        :return:A new DataPreprocessor, incorporating the previous steps plus the new denoise
        """
        return _DataDenoiser(self, smooth, patch_radius)

    def normalise(self) -> DataPreprocessor:
        """
        Normalize DWI Data based on b0 image.

        The weights are divided by their b0 value.

        :return: A new DataPreprocessor, incorporating the previous steps plus the new normalise
        """
        return _DataNormaliser(self)

    def crop(self, b_value=1000.0, max_deviation=100.0) -> DataPreprocessor:
        """
        Crops the dataset based on B-value.

        All measurements where the B-value deviates more than `max_deviation` from the `b_value`
        are removed from the dataset.

        :param b_value: the intended B-value
        :param max_deviation: the maximum allowed deviation
        :return: A new DataPreprocessor, incorporating the previous steps plus the new crop
        """
        return _DataCropper(self, b_value, max_deviation)

    def fa_estimate(self):
        """
        Does the FA estimation at the current position in the pipeline
        :return: A new DataPreprocessor, incorporating the previous steps plus the new fa estimate
        """
        return _DataFAEstimator(self)

    def get_hcp(self, path: str, b0_threshold: float = 10.0) -> DataContainer:
        """
        Loads a HCP Dataset and preprocesses it, returning a DataContainer

        :param path: The path of the HCP Dataset
        :param b0_threshold: The threshold for the b0 image
        :return: A newly created DataContainer with the preprocessed HCP data.
        """
        file_mapping = {'bvals': 'bvals', 'bvecs': 'bvecs', 'img': 'data.nii.gz',
                        't1': 'T1w_acpc_dc_restore_1.25.nii.gz', 'mask': 'nodif_brain_mask.nii.gz'}
        return self._get_from_file_mapping(path, file_mapping, b0_threshold)

    def get_ismrm(self, path: str, b0_threshold: float = 10.0) -> DataContainer:
        """
        Loads an ISMRM Dataset and preprocesses it, returning a DataContainer

        :param path: The path of the HCP Dataset
        :param b0_threshold: The threshold for the b0 image
        :return: A newly created DataContainer with the preprocessed ISMRM data.
        """
        file_mapping = {'bvals': 'Diffusion.bvals', 'bvecs': 'Diffusion.bvecs',
                        'img': 'Diffusion.nii.gz', 't1': 'T1.nii.gz'}
        return self._get_from_file_mapping(path, file_mapping, b0_threshold)

    def _get_from_file_mapping(self, path, file_mapping: dict, b0_threshold: float = 10.0):

        path_mapping = {key: os.path.join(path, file_mapping[key]) for key in file_mapping}
        bvals, bvecs = read_bvals_bvecs(path_mapping['bvals'],
                                        path_mapping['bvecs'])

        # img, t1, gradient table, affine and dwi
        img = nb.load(path_mapping['img'])
        t1 = nb.load(path_mapping['t1']).get_data()

        dwi = img.get_data().astype("float32")

        aff = img.affine

        # binary mask
        if 'mask' in path_mapping:
            binary_mask = nb.load(path_mapping['mask']).get_data()
        else:
            _, binary_mask = median_otsu(dwi[..., 0], 2, 1)

        # calculating b0
        b0 = dwi[..., bvals < b0_threshold].mean(axis=-1)

        # Do not generate fa yet
        fa = None

        data_container = DataContainer(bvals, bvecs, t1, dwi, aff, binary_mask, b0, fa)
        return self._preprocess(data_container)


class _DataCropper(DataPreprocessor):
    def __init__(self, parent, b_value, max_deviation):
        super().__init__(parent)
        self.b_value = b_value
        self.max_deviation = max_deviation

    def _preprocess(self, data_container: DataContainer) -> DataContainer:
        bvals, bvecs, t1, dwi, aff, binary_mask, b0, fa = \
            super()._preprocess(data_container)

        indices = np.where(np.abs(bvals - self.b_value) < self.max_deviation)[0]

        dwi = dwi[..., indices]
        bvals = bvals[indices]
        bvecs = bvecs[indices]

        return DataContainer(bvals, bvecs, t1, dwi, aff, binary_mask, b0, fa)

    def __str__(self):
        return str(self._parent) + "-Crop-b_value-{}-deviation-{}".format(self.b_value, self.max_deviation)


class _DataNormaliser(DataPreprocessor):
    def __init__(self, parent):
        super().__init__(parent)

    def _preprocess(self, data_container: DataContainer) -> DataContainer:
        bvals, bvecs, t1, dwi, aff, binary_mask, b0, fa = \
            super()._preprocess(data_container)

        b0 = b0[..., None]

        nb_erroneous_voxels = np.sum(dwi > b0)
        if nb_erroneous_voxels != 0:
            dwi = np.minimum(dwi, b0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dwi = dwi / b0
            dwi[np.logical_not(np.isfinite(dwi))] = 0.

        return DataContainer(bvals, bvecs, t1, dwi, aff, binary_mask, b0, fa)

    def __str__(self):
        return str(self._parent) + "-Normalise"


class _DataDenoiser(DataPreprocessor):
    def __init__(self, parent, smooth, patch_radius):
        super().__init__(parent)
        self.smooth = smooth
        self.patch_radius = patch_radius

    def _preprocess(self, data_container: DataContainer) -> DataContainer:
        bvals, bvecs, t1, dwi, aff, binary_mask, b0, fa = \
            super()._preprocess(data_container)
        gtab = gradient_table(bvals, bvecs)
        sigma = pca_noise_estimate(dwi, gtab, correct_bias=True,
                                   smooth=self.smooth)
        dwi = localpca(dwi, sigma=sigma,
                       patch_radius=self.patch_radius)
        return DataContainer(bvals, bvecs, t1, dwi, aff, binary_mask, b0, fa)

    def __str__(self) -> str:
        return str(self._parent) + "-Denoise-smooth-{}-patch_radius-{}".format(self.smooth, self.patch_radius)


class _DataFAEstimator(DataPreprocessor):
    def __init__(self, parent):
        super().__init__(parent)

    def _preprocess(self, data_container: DataContainer) -> DataContainer:
        bvals, bvecs, t1, dwi, aff, binary_mask, b0, fa = \
            super()._preprocess(data_container)

        # calculating fractional anisotropy (fa)
        gtab = gradient_table(bvals, bvecs)
        dti_model = dti.TensorModel(gtab, fit_method='LS')
        dti_fit = dti_model.fit(dwi)
        fa = dti_fit.fa
        return DataContainer(bvals, bvecs, t1, dwi, aff, binary_mask, b0, fa)

    def __str__(self):
        return str(self._parent) + "-FA-estimation"


class DataContainer(collections.namedtuple("DataContainer", "bvals, bvecs, t1, dwi, aff, binary_mask, b0, fa")):

    def __new__(cls, bvals: np.ndarray, bvecs: np.ndarray, t1: np.ndarray,
                dwi: np.ndarray, aff: np.ndarray, binary_mask: np.ndarray, b0: np.ndarray, fa: np.ndarray | None):

        self = super().__new__(cls, bvals, bvecs, t1, dwi, aff,
                               binary_mask, b0, fa)

        x_range = np.arange(dwi.shape[0])
        y_range = np.arange(dwi.shape[1])
        z_range = np.arange(dwi.shape[2])

        self.interpolator = RegularGridInterpolator((x_range, y_range, z_range), dwi)
        return self

    def to_ijk(self, points: np.ndarray) -> np.ndarray:
        """
        Converts given RAS+ points to IJK in DataContainers Image Coordinates.

        The conversion happens using the affine of the DWI image.
        It should be noted that the dimension of the given point array stays the same.

        :param points: The points to convert.
        :return: The converted points.
        """

        aff = np.linalg.inv(self.aff)
        return apply_affine(aff, points)

    def to_ras(self, points: np.ndarray) -> np.ndarray:
        """
        Converts given IJK points in DataContainers Coordinate System to RAS+.

        The conversion happens using the affine of the DWI image.
        It should be noted that the dimension of the given point array stays the same.

        :param points:  The points to convert.
        :return: The converted points.
        """
        return apply_affine(self.aff, points)

    def get_interpolated_dwi(self, points: np.ndarray, postprocessing: Callable = None,
                             ignore_outside_points: bool = False) -> np.ndarray:
        """
        Returns interpolated dwi for given RAS+ points.

        The shape of the input points will be retained for the return array,
        only the last dimension will be changed from 3 to the (interpolated) DWI-size accordingly.

        If you provide a postprocessing method, the interpolated data is then fed through this postprocessing option.

        :param numpy.ndarray points: The array containing the points. Shape is matched in output.
        :param data.postprocessing postprocessing:
        :param ignore_outside_points: A postprocessing method, e.g res100, raw, spherical_harmonics etc.
            which will be applied to the output.
        :return: The DWI-Values interpolated for the given points. The input shape is matched aside of
        the last dimension.
        :type: numpy.ndarray

        """
        new_shape = (*points.shape[:-1], -1)

        points = self.to_ijk(points).reshape(-1, 3)

        is_outside = ((points[:, 0] < 0) + (points[:, 0] >= self.dwi.shape[0]) +  # OR
                      (points[:, 1] < 0) + (points[:, 1] >= self.dwi.shape[1]) +
                      (points[:, 2] < 0) + (points[:, 2] >= self.dwi.shape[2])) > 0

        if np.sum(is_outside) > 0 and not ignore_outside_points:
            raise PointOutsideOfDWIError(self, self.to_ras(points), self.to_ras(points[is_outside]))

        result = np.empty_like(points)

        result[not is_outside] = self.interpolator(points[not is_outside])

        if postprocessing is not None:
            result[not is_outside] = postprocessing(result[not is_outside], self.b0,
                                                    self.bvecs,
                                                    self.bvals)
        result[is_outside, :] = 0
        result = result.reshape(new_shape)
        return result

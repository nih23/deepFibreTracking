# pylint: disable=attribute-defined-outside-init
"""TODO write documentation
"""
import os
import warnings

import torch
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.align.reslice import reslice
from dipy.segment.mask import median_otsu
from dipy.align.imaffine import interpolate_scalar_3d


import numpy as np
import nibabel as nb
from nibabel.affines import apply_affine

from src.config import Config

class Object():
    """Just a plain object usable to store information"""

class Error(Exception):
    """Base class for Data exceptions."""

    def __init__(self, msg=''):
        self.message = msg
        Exception.__init__(self, msg)

    def __repr__(self):
        return self.message

    __str__ = __repr__

class DeviceNotRetrievableError(Error):
    """Error thrown if get_device is called on CUDA tensor."""

    def __init__(self, device):
        self.device = device
        Error.__init__(self, msg=("get_device() can't be called on non-CUDA Tensors."
                                  "Current device: {}".format(device)))

class DataContainerNotLoadableError(Error):
    """Error thrown if DataContainer is unable to load specified files"""

    def __init__(self, path, file):
        self.path = path
        self.file = file
        Error.__init__(self, msg=("The File '{file}' "
                                  "can't be retrieved from folder '{path}' for the dataset.")
                       .format(file=file, path=path))

class PointOutsideOfDWIError(Error):
    """Error thrown if given are outside of the DWI-Image"""

    def __init__(self, data_container, points, affected_points):
        self.data_container = data_container
        self.points = points
        self.affected_points = affected_points
        Error.__init__(self, msg=("While parsing {no_points} points for further processing, "
                                  "it became apparent that {aff} of the points "
                                  "doesn't lay inside of DataContainer '{id}'.")
                       .format(no_points=points.size, id=data_container.id, aff=affected_points))

class DWIAlreadyCroppedError(Error):
    """Error thrown if given are outside of the DWI-Image"""

    def __init__(self, data_container, bval, dev):
        self.data_container = data_container
        self.bval = bval
        self.max_deviation = dev
        Error.__init__(self, msg=("The dataset {id} is already cropped with b_value "
                                  "{bval} and deviation {dev}.")
                       .format(id=data_container.id, bval=bval, dev=dev))

class MovableData():
    """The movable data class - make tensor based classes easy movable.
       Tensors musst be attributes of the class and not nested."""
    device = None
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cpu")
        self.device = device

    def _get_tensors(self):
        """Returns all retrieved tensors of class"""
        tensors = {}
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
        return tensors

    def _set_tensor(self, attribute, tensor):
        """Sets tensor with key from _get_tensors()"""
        setattr(self, attribute, tensor)

    def cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        """Moves all Tensors to specified CUDA device"""
        for attribute, tensor in self._get_tensors().items():
            cuda_tensor = tensor.cuda(device=device, non_blocking=non_blocking,
                                      memory_format=memory_format)
            self._set_tensor(attribute, cuda_tensor)
            self.device = cuda_tensor.device
        return self

    def cpu(self, memory_format=torch.preserve_format):
        """Moves all Tensors to CPU"""
        for attribute, tensor in self._get_tensors().items():
            cpu_tensor = tensor.cpu(memory_format=memory_format)
            self._set_tensor(attribute, cpu_tensor)
        self.device = torch.device('cpu')
        return self

    def to(self, *args, **kwargs):
        """Moves all tensors to specified device"""
        for attribute, tensor in self._get_tensors().items():
            tensor = tensor.to(*args, **kwargs)
            self._set_tensor(attribute, tensor)
            self.device = tensor.device
        return self

    def get_device(self):
        """Returns CUDA device number. Throwns a DeviceNotRetrievableError if on CPU."""
        if self.device.type == "cpu":
            raise DeviceNotRetrievableError(self.device)
        return self.device.index

class DataContainer():
    """The DataContainer class representing a single dataset"""

    def __init__(self, path, file_names, denoise=None, b0_threshold=None):
        if denoise is None:
            denoise = Config.get_config().getboolean("data", "denoise", fallback="no")
        if b0_threshold is None:
            b0_threshold = Config.get_config().getfloat("data", "b0-threshold", fallback="10")
        self.options = Object()
        self.options.denoised = denoise
        self.options.cropped = False
        self.options.b0_threshold = b0_threshold
        self.path = path.rstrip(os.path.sep) + os.path.sep
        self.data = self._retrieve_data(file_names, denoise=denoise, b0_threshold=b0_threshold)
        self.id = ("DataContainer" + self.path.replace(os.path.sep, "-") +
                   "b0thr-" + str(b0_threshold))
        if self.options.denoised:
            self.id = self.id + "-denoised"

    def _retrieve_data(self, file_names, denoise=False, b0_threshold=None):
        """Reads data from files and saves them into self.data"""
        data = Object()
        try:
            data.bvals, data.bvecs = read_bvals_bvecs(self.path + file_names['bvals'],
                                                      self.path + file_names['bvecs'])
            data.img = nb.load(self.path + file_names['img'])
            data.t1 = nb.load(self.path + file_names['t1']).get_data()
        except FileNotFoundError as error:
            raise DataContainerNotLoadableError(self.path, error.filename) from None

        data.gtab = gradient_table(bvals=data.bvals, bvecs=data.bvecs)
        data.dwi = data.img.get_data().astype("float32")
        data.aff = data.img.affine

        if denoise:
            sigma = pca_noise_estimate(data.dwi, data.gtab, correct_bias=True,
                                       smooth=Config.get_config().getint("denoise", "smooth",
                                                                         fallback="3"))
            data.dwi = localpca(data.dwi, sigma=sigma,
                                patch_radius=Config.get_config().getint("denoise", "pathRadius",
                                                                        fallback="2"))
        if 'mask' in file_names:
            data.binarymask = nb.load(self.path + file_names['mask']).get_data()
        else:
            _, data.binarymask = median_otsu(data.dwi[..., 0], 2, 1)

        data.b0 = data.dwi[..., data.bvals < b0_threshold].mean(axis=-1)

        return data

    def to_ijk(self, points):
        """Converts given RAS+ points to IJK in DataContainers Image Coordinates"""
        aff = np.linalg.inv(self.data.aff)
        return apply_affine(aff, points)

    def to_ras(self, points):
        """Converts given IJK points in DataContainers Coordinate System to RAS+"""
        aff = self.data.aff
        return apply_affine(aff, points)

    def get_interpolated_dwi(self, points, ignore_outside_points=False):
        """Returns interpolated dwi for given RAS+ points.
        If ignore_outside_points is set to true,
        no error will be thrown for points outside of the image"""
        points = self.to_ijk(points)
        shape = points.shape
        new_shape = (*shape[:-1], self.data.dwi.shape[-1])
        result = np.zeros(new_shape)
        for i in range(self.data.dwi.shape[-1]):
            (out, inside) = interpolate_scalar_3d(self.data.dwi[..., i], points.reshape(-1, 3))
            if np.any(inside == 0) and not ignore_outside_points:
                raise PointOutsideOfDWIError(self, points, np.sum(inside == 0))
            result[..., i] = out.reshape((*new_shape[:-1]))
        return result

    def crop(self, b_value=None, max_deviation=None):
        """Crop the dataset based on B-value"""
        if self.options.cropped:
            raise DWIAlreadyCroppedError(self, self.options.crop_b, self.options.crop_max_deviation)
        if b_value is None:
            b_value = Config.get_config().getfloat("data", "cropB-Value", fallback="1000.0")
        if max_deviation is None:
            max_deviation = Config.get_config().getfloat("data", "cropMaxDeviation", fallback="100")

        indices = np.where(np.abs(self.data.bvals - b_value) < max_deviation)[0]
        self.data.dwi = self.data.dwi[..., indices]
        self.data.bvals = self.data.bvals[indices]
        self.data.bvecs = self.data.bvecs[indices]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.data.gtab = gradient_table(bvals=self.data.bvals, bvecs=self.data.bvecs)

        self.options.cropped = True
        self.options.crop_b = b_value
        self.options.crop_max_deviation = max_deviation
        self.id = self.id + "-cropped[{b}, {dev}]".format(b=b_value, dev=max_deviation)
        return self

    def normalize(self):
        """Normalize DWI Data based on b0 image. It is recommendable to crop the dataset first"""
        b0 = self.data.b0[..., None]

        nb_erroneous_voxels = np.sum(self.data.dwi > b0)
        if nb_erroneous_voxels != 0:
            weights = np.minimum(self.data.dwi, b0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weights = weights / b0
            weights[np.logical_not(np.isfinite(weights))] = 0.

        self.data.dwi = weights
        self.id = self.id + "-normalized"
        return self


class HCPDataContainer(DataContainer):
    """The container for HCPData"""

    def __init__(self, hcpid, denoise=None, b0_threshold=None):
        path = Config.get_config().get("data", "pathHCP", fallback='data/HCP/{id}').format(id=hcpid)
        self.hcp_id = hcpid
        paths = {'bvals':'bvals', 'bvecs':'bvecs', 'img':'data.nii.gz',
                 't1':'T1w_acpc_dc_restore_1.25.nii.gz', 'mask':'nodif_brain_mask.nii.gz'}
        DataContainer.__init__(self, path, paths, denoise=denoise, b0_threshold=b0_threshold)
        self.id = ("HCPDataContainer-HCP{id}-b0thr-{b0}"
                   .format(id=self.hcp_id, b0=self.options.b0_threshold))
        if self.options.denoised:
            self.id = self.id + "-denoised"

class ISMRMDataContainer(DataContainer):
    """The container for ISMRM2015 Data"""
    def __init__(self, denoise=None, rescale_to_hcp=None, b0_threshold=None):
        path = Config.get_config().get("data", "pathISMRM", fallback='data/ISMRM2015')
        paths = {'bvals':'Diffusion.bvals', 'bvecs':'Diffusion.bvecs',
                 'img':'Diffusion.nii.gz', 't1':'T1.nii.gz'}
        DataContainer.__init__(self, path, paths, denoise=denoise, b0_threshold=b0_threshold)
        if rescale_to_hcp is None:
            rescale_to_hcp = Config.get_config().getboolean("data", "rescaleHCPData", fallback="no")
        self.options.rescale_to_hcp = rescale_to_hcp

        self.id = "ISMRMDataContainer-b0thr-{b0}".format(b0=self.options.b0_threshold)
        if self.options.denoised:
            self.id = self.id + "-denoised"
        if rescale_to_hcp:
            self._rescale_to_hcp()
            self.id = self.id + "-rescaled"

    def _rescale_to_hcp(self):
        """Rescales the ISMRM Dataset to HCP Coordinates"""
        data = self.data
        zooms = data.img.header.get_zooms()[:3]
        new_zooms = (1.25, 1.25, 1.25) # similar to HCP
        data.dwi, data.aff = reslice(data.dwi, data.aff, zooms, new_zooms)
        self.data = data


class TypeClass:
    """A class representing a type"""
    def __init__(self):
        self.type = None

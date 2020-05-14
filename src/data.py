# pylint: disable=attribute-defined-outside-init
"""class:

MovableData
    A class representing a class containg movable tensors.

    methods:

        MovableData(device=None)
            Creates a MovableDataset on specified device. Default: cpu

        cuda(device=None, non_blocking=False, memory_format=torch.preserve_format)
            Moves every Tensor in MovableData to CUDA device (with params)

        cpu(self, memory_format=torch.preserve_format)
            Moves every Tensor in MovableData to cpu device (with params)

        to(*args, **kwargs)
            Moves every Tensor in MovableData to specified device

        get_device()
            If CUDA device, returns the current device number, for example 0 in case of cuda:0
            Throws a DeviceNotRetrievableError if device is CPU

    attributes:

        device
            A torch.device, representing the current device.

DataContainer
    An instance is representing a DWI-Dataset

    methods:

        DataContainer(path, filenames, denoise=None)
            Creates a DataContainer while retrieving files with filenames from path
            If denoise is specified, the data will either be denoised or not.
            Else, the value saved in the configuration will be chosen.

        to_ijk(self, points)
            Converts given RAS+ points to IJK in DataContainers Image Coordinates

        to_ras(self, points)
            Converts given IJK points in DataContainers Coordinate System to RAS+

        get_interpolated_dwi(self, points, ignore_outside_points=False)
            Returns interpolated dwi for given RAS+ points in the same data shape.
            If ignore_outside_points is set to true,
            no error will be thrown for points outside of the image

    attributes:

        id
            A unique identifier of the loaded dataset

        path
            The path from which the dataset was retrieved.

        data
            The loaded data as a tuple (bvals, bvecs, gtab, dwi, aff, t1, img).

        is_denoised
            A boolean indicating wether denoising happened or not.

HCPDataContainer
    Represents any HCP Dataset

    methods:

        HCPDataContainer(id, denoise=None)
            Loads HCP-Data with specified ID. Path is retrieved from config.
            If denoise is specified, the data will either be denoised or not.
            Else, the value saved in the configuration will be chosen.

    attributes:

        id
            A unique identifier of the loaded dataset

        path
            The path from which the dataset was retrieved.

        data
            The loaded data as an object.

        is_denoised
            A boolean indicating wether denoising happened or not.

        hcp_id
            The HCP-ID of the loaded dataset

ISMRMDataContainer
    Represents the ISMRM 2015 Dataset

    methods:

        ISMRMDataContainer(denoise=None, rescale_to_hcp=None)
            Loads ISMRMData, path is specified in config.
            If denoise is specified, the data will either be denoised or not.
            Else, the value saved in the configuration will be chosen.
            The same applies to the rescale_to_hcp option,
            which will rescale the dwi data to 1.25 mm³ measures.

    attributes:

        id
            A unique identifier of the loaded dataset

        path
            The path from which the dataset was retrieved.

        data
            The loaded data as an object.

        is_denoised
            A boolean indicating wether denoising happened or not.

        is_rescaled
            A boolean indicating wether the data is rescaled or not.

"""
import os

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

    def __init__(self, path, file_names, denoise=None):
        if denoise is None:
            denoise = Config.get_config().getboolean("data", "denoise", fallback="no")
        self.is_denoised = denoise
        self.path = path.rstrip(os.path.sep) + os.path.sep
        self.data = self._retrieve_data(file_names, denoise=denoise)
        self.id = "DataContainer"+ self.path.replace(os.path.sep, "-").rstrip("-")
        if self.is_denoised:
            self.id = self.id + "-denoised"

    def _retrieve_data(self, file_names, denoise=False):
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
        data.dwi = data.img.get_data()
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
            _, data.binarymask = median_otsu(data.dwi[:, :, :, 0], 2, 1)

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
            result[..., i] = out.reshape((*new_shape, 1))
        return result

class HCPDataContainer(DataContainer):
    """The container for HCPData"""

    def __init__(self, hcpid, denoise=None):
        path = Config.get_config().get("data", "pathHCP", fallback='data/HCP/{id}').format(id=hcpid)
        self.hcp_id = hcpid
        paths = {'bvals':'bvals', 'bvecs':'bvecs', 'img':'data.nii.gz',
                 't1':'T1w_acpc_dc_restore_1.25.nii.gz', 'mask':'nodif_brain_mask.nii.gz'}
        DataContainer.__init__(self, path, paths, denoise=denoise)
        self.id = "HCPDataContainer-HCP{id}".format(id=self.hcp_id)
        if self.is_denoised:
            self.id = self.id + "-denoised"

class ISMRMDataContainer(DataContainer):
    """The container for ISMRM2015 Data"""
    def __init__(self, denoise=None, rescale_to_hcp=None):
        path = Config.get_config().get("data", "pathISMRM", fallback='data/ISMRM2015')
        paths = {'bvals':'Diffusion.bvals', 'bvecs':'Diffusion.bvecs',
                 'img':'ismrm_denoised_preproc_mrtrix.nii.gz', 't1':'T1.nii.gz'}
        DataContainer.__init__(self, path, paths, denoise=denoise)
        if rescale_to_hcp is None:
            rescale_to_hcp = Config.get_config().getboolean("data", "rescaleHCPData", fallback="no")
        self.is_rescaled = rescale_to_hcp

        self.id = "ISMRMDataContainer"
        if self.is_denoised:
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

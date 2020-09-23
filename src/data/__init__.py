# pylint: disable=attribute-defined-outside-init
"""
The data module is handling any kind of DWI-Data.

Available subpackages
---------------------
postprocessing
    Provides multiple postprocessing options for raw dwi data.

Classes
-------
Error
    Base class for Data exceptions.
DeviceNotRetrievableError
    Exception thrown if get_device is called on non-CUDA tensor.
DataContainerNotLoadableError
    Exception thrown if DataContainer is unable to load specified files.
PointOutsideOfDWIError
    Error thrown if given points are outside of the DWI-Image.
DWIAlreadyCroppedError
    Error thrown if the DWI data should be cropped multiple times.
MovableData
    Class can be used to make classes handling multiple tensors more easily movable.
DataContainer
    DataContainer is representing a single DWI Image.
HCPDataContainer
    DataContainer representing a HCP Image
ISMRMDataContainer
    DataContainer representing ISMRM Image.
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
import dipy.reconst.dti as dti

import numpy as np
import nibabel as nb
from nibabel.affines import apply_affine

from src.config import Config
class Error(Exception):
    """
    Base class for Data exceptions.

    Every Error happening from code of this class will inherit this one.
    The single parameter `msg` represents the error representing message.

    This class can be used to filter the exceptions for data exceptions.

    Attributes
    ----------
    message: str, optional
        The message given is stored here.

    Examples
    --------

    >>> e = Error(msg='Just a sample message')
    >>> raise e from None
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    src.data.Error: Just a sample message
    """

    def __init__(self, msg=''):
        """
        Parameters
        ----------
        msg : str
            The message which accompanying the error, by default ''.
        """
        self.message = msg
        Exception.__init__(self, msg)

    def __repr__(self):
        return self.message

    __str__ = __repr__ # simplify stringify behaviour

class DeviceNotRetrievableError(Error):
    """
    Exception thrown if get_device is called on non-CUDA tensor.

    There is only one CPU usable for active workload. Therefore,
    no cpu number is specified.

    Attributes
    ----------
    message: str
        The error message is stored here.
    device:
        The device currently active.

    Examples
    --------
    The error class can be initialized with the following:

    >>> import torch
    >>> a = DeviceNotRetrievableError(torch.device('cpu'))
    >>> raise a from None
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    src.data.DeviceNotRetrievableError: get_device() can't be called on non-CUDA Tensors.
                                        Current device: cpu

    A common mistake, on which the exception could be raised, would be the following:

    >>> a = MovableData()
    >>> a.get_device()
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "/home/jos/deepFibreTracking/src/data/__init__.py", line 165, in get_device

    src.data.DeviceNotRetrievableError: get_device() can't be called on non-CUDA Tensors.
    Current device: cpu
    """

    def __init__(self, device):
        """
        Parameters
        ----------
        device: torch.device
            The current device on which the error occured.
        """
        self.device = device
        Error.__init__(self, msg=("get_device() can't be called on non-CUDA Tensors. "
                                  "Current device: {}".format(device)))

class DataContainerNotLoadableError(Error):
    """
    Exception thrown if DataContainer is unable to load specified files.

    After initializing a DataContainer, it looks for defined files in given folder.
    If the software is unable to find a concrete file, this exception is thrown.

    Attributes
    ----------
    message: str
        The error message is stored here.
    path: str
        The path in which the software was unable to find the file.
    file: str
        The filename which couldn't be found in folder.

    Examples
    --------
    An example initiation of the error:

    >>> a = DataContainerNotLoadableError("path/to/folder", "notexisting.txt")
    >>> raise a from None
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    src.data.DataContainerNotLoadableError: The File 'notexisting.txt' \
        can't be retrieved from folder 'path/to/folder' for the dataset.
    """

    def __init__(self, path, file):
        """
        Parameters
        ----------
        path: str
            The path in which the software was unable to find the file.
        file: str
            The filename which couldn't be found in folder.
        """
        self.path = path
        self.file = file
        Error.__init__(self, msg=("The File '{file}' "
                                  "can't be retrieved from folder '{path}' for the dataset.")
                       .format(file=file, path=path))

class PointOutsideOfDWIError(Error):
    """
    Error thrown if given points are outside of the DWI-Image.

    This can be bypassed by passing `ignore_outside_points = True`
    to the raising function. However, it should be noted that this
    is not recommendable behaviour.

    Attributes
    ----------
    data_container : DataContainer
        The `DataContainer` whose DWI-Image is too small to cover the points.
    points: ndarray
        The point array which is responsible for raising the error.
    affected_points: ndarray
        The affected points beingn outside of the DWI-image.
    """

    def __init__(self, data_container, points, affected_points):
        """
        Parameters
        ----------
        data_container : DataContainer
            The `DataContainer` whose DWI-Image is too small to cover the points.
        points: ndarray
            The point array which is responsible for raising the error.
        affected_points: ndarray
            The affected points beingn outside of the DWI-image.
        """
        self.data_container = data_container
        self.points = points
        self.affected_points = affected_points
        Error.__init__(self, msg=("While parsing {no_points} points for further processing, "
                                  "it became apparent that {aff} of the points "
                                  "doesn't lay inside of DataContainer '{id}'.")
                       .format(no_points=points.size, id=data_container.id, aff=affected_points))

class DWIAlreadyCroppedError(Error):
    """
    Error thrown if the DWI data should be cropped multiple times.

    The cropping of DWI is not reversable in `DataContainer`. Therefore,
    `dc.crop(*args)` doesn't necessarily equal `dc.crop(*other_args).crop(*args)`.
    To prevent this potential confusing behaviour, this exception will be thrown on the latter.

    Attributes
    ----------
    data_container : DataContainer
        The affected `DataContainer`.
    bval: float
        The b-value used for the first, real cropping of the `DataContainer`.
    max_deviation: float
        The maximum deviation allowed while cropping.
    """

    def __init__(self, data_container, bval, dev):
        """
        Parameters
        ----------
        data_container : DataContainer
            The affected `DataContainer`.
        bval: float
            The b-value used for the first, real cropping of the `DataContainer`.
        dev: float
            The maximum deviation allowed while cropping.
        """
        self.data_container = data_container
        self.bval = bval
        self.max_deviation = dev
        Error.__init__(self, msg=("The dataset {id} is already cropped with b_value "
                                  "{bval} and deviation {dev}.")
                       .format(id=data_container.id, bval=bval, dev=dev))

class DWIAlreadyNormalizedError(Error):
    """Error thrown if DWI normalize function is getting called multiple times.

    You have to recreate a new DataContainer if you want to change the normalization

    Attributes
    ----------
    data_container : DataContainer
        The affected `DataContainer`.
    """
    def __init__(self, data_container):
        """Parameters
        ----------
        data_container : DataContainer
            The DataContainer which is already normalized.    
        """

        self.data_container = data_container
        Error.__init__(self, msg="The DWI of the DataContainer {id} is already normalized. ".format(id=data_container.id))

class DWINotNormalizedError(Error):
    """Error thrown if DWI has to be normalized for calling this function.

    Call DataContainer.normalize() to normalize.

    Attributes
    ----------
    data_container : DataContainer
        The affected `DataContainer`.
    """
    def __init__(self, data_container):
        """Parameters
        ----------
        data_container : DataContainer
            The DataContainer which is already normalized.    
        """

        self.data_container = data_container
        Error.__init__(self, msg="The DWI of the DataContainer {id} isn't normalized yet. Call .normalize() first!".format(id=data_container.id))

class MovableData():
    """
    This class can be used to make classes handling multiple tensors more easily movable.

    With simple inheritance, all of those must be instances of `torch.Tensor` or `MovableData`.
    Also, they have to be direct attributes of the object and are not allowed to be nested.

    Attributes
    ----------
    device: torch.device, optional
        The device the movable data currently is located on.

    Methods
    -------
    cuda(device=None, non_blocking=False, memory_format=torch.preserve_format)
        Moves the MovableData to specified or default CUDA device.
    cpu(memory_format=torch.preserve_format)
        Moves the MovableData to cpu.
    to(*args, **kwargs)
        Moves the MovableData to specified device.
        See `torch.Tensor.to(...)` for more details on usage.
    get_device()
        Returns the CUDA device number if possible. Raises `DeviceNotRetrievableError` otherwise.

    Inheritance
    -----------
    To modify and inherit the `MovableData` class, overwrite the following functions:

    _get_tensors()
        This should return all `torch.Tensor` and `MovableData` instances of your class,
        in a key value pair `dict`.

    _set_tensor(key, tensor)
        This should replace the reference to the tensor with given key with the new, moved tensor.

    If those two methods are properly inherited, the visible functions should work as normal.
    If you plan on add other class types to the `_get_tensors` method, make sure that they implement
    the cuda, cpu, to and get_device methods in the same manner as `torch.Tensor` instances.
    """
    device = None
    def __init__(self, device=None):
        """
        Parameters
        ----------
        device : torch.device, optional
            The device which the `MovableData` should be moved to on load, by default cpu.
        """
        if device is None:
            device = torch.device("cpu")
        self.device = device

    def _get_tensors(self):
        """
        Returns a dict containing all `torch.Tensor` and `MovableData` instances
        and their assigned keys.

        The default implementation searches for those on the attribute level.
        If your child class contains tensors at other positions, it is recommendable to
        overwrite this function and the `_set_tensor` function.

        Returns
        -------
        dict
            The dict containing every `torch.Tensor` and `MovableData` with their assigned keys.

        See Also
        --------
        _set_tensor: implementations depend on each other
        """
        tensors = {}
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor) or isinstance(value, MovableData):
                tensors[key] = value
        return tensors

    def _set_tensor(self, key, tensor):
        """
        Sets the tensor with the assigned key to his value.

        In the default implementation, this works analogously to `_get_tensors`:
        It sets the attribute with the name key to the given object/tensor.
        If your child class contains tensors at other positions, it is recommendable to
        overwrite this function and the `_get_tensors` function.

        Parameters
        ----------
        key : str
            The key of the original tensor.
        tensor : object
            The new tensor which should replace the original one.

        See Also
        --------
        _get_tensors: implementations depend on each other
        """
        setattr(self, key, tensor)

    def cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        """
        Returns this object in CUDA memory.

        If this object is already in CUDA memory and on the correct device,
        then no movement is performed and the original object is returned.

        Parameters
        ----------
        device : `torch.device`, optional
            The destination GPU device. Defaults to the current CUDA device.
        non_blocking : `bool`, optional
             If `True` and the source is in pinned memory, the copy will be asynchronous with
             respect to the host. Otherwise, the argument has no effect, by default `False`.
        memory_format : `torch.memory_format`, optional
            the desired memory format of returned Tensor, by default `torch.preserve_format`.

        Returns
        -------
        MovableData
            The object moved to specified device
        """
        for attribute, tensor in self._get_tensors().items():
            cuda_tensor = tensor.cuda(device=device, non_blocking=non_blocking,
                                      memory_format=memory_format)
            self._set_tensor(attribute, cuda_tensor)
            self.device = cuda_tensor.device
        return self

    def cpu(self, memory_format=torch.preserve_format):
        """
        Returns a copy of this object in CPU memory.

        If this object is already in CPU memory and on the correct device,
        then no copy is performed and the original object is returned.

        Parameters
        ----------
        memory_format : `torch.memory_format`, optional
            the desired memory format of returned Tensor, by default `torch.preserve_format`.

        Returns
        -------
        MovableData
            The object moved to specified device
        """
        for attribute, tensor in self._get_tensors().items():
            cpu_tensor = tensor.cpu(memory_format=memory_format)
            self._set_tensor(attribute, cpu_tensor)
        self.device = torch.device('cpu')
        return self

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion.
        A `torch.dtype` and `torch.device` are inferred from the arguments of
        `self.to(*args, **kwargs)`.

        Here are the ways to call `to`:

        to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
            Returns MovableData with specified `dtype`

        to(device=None, dtype=None, non_blocking=False, copy=False,
        memory_format=torch.preserve_format) -> Tensor
            Returns MovableData on specified `device`

        to(other, non_blocking=False, copy=False) → Tensor
            Returns MovableData with same `dtype` and `device` as `other`
        Returns
        -------
        MovableData
            The object moved to specified device
        """
        for attribute, tensor in self._get_tensors().items():
            tensor = tensor.to(*args, **kwargs)
            self._set_tensor(attribute, tensor)
            self.device = tensor.device
        return self

    def get_device(self):
        """
        For CUDA tensors, this function returns the device ordinal of the GPU on which the tensor
        resides. For CPU tensors, an error is thrown.

        Returns
        -------
        int
            The device ordinal

        Raises
        ------
        DeviceNotRetrievableError
            This description is thrown if the tensor is currently on the cpu,
            therefore, no device ordinal exists.
        """
        if self.device.type == "cpu":
            raise DeviceNotRetrievableError(self.device)
        return self.device.index

class DataContainer():
    """
    The DataContainer class is representing a single DWI Dataset.

    It contains basic functions to work with the data.
    The data itself is accessable in the `self.data` attribute.

    The `self.data` attribute contains the following
        - bvals: the B-values
        - bvecs: the B-vectors matching the bvals
        - img: the DWI-Image file
        - t1: the T1-File data
        - gtab: the calculated gradient table
        - dwi: the real DWI data
        - aff: the affine used for coordinate transformation
        - binarymask: a binarymask usable to separate brain from the rest
        - b0: the b0 image usable for normalization etc.

    Attributes
    ----------
    options: SimpleNamespace
        The configuration of the current DWI.
    path: str
        The path of the loaded DWI-Data.
    data: SimpleNamespace
        The dwi data, referenced in the SimpleNamespace's attributes.
    id: str
        An identifier of the current DWI-Data including its preprocessing.

    Methods
    -------
    to_ijk(points)
        Returns conversion of given RAS+ points into IJK format for DWI-File.
    to_ras(points)
        Returns conversion of given IJK points for DWI-File into RAS+ format.
    get_interpolated_dwi(points, ignore_outside_points=False)
        Returns 3D-interpolated DWI-Image values at the given RAS+ points.
    crop(b_value=None, max_deviation=None)
        Crops DWI-Data to given b_value and deviation. If param equals `None`,
        the values specified in the configuration file are used.
        Returns `self`.
    normalize()
        Normalizes the DWI-Image based on b0-Image.
        Returns `self`.
    get_fa()
        Generates the FA based on eigenvalues and returns them.
    Inheritance
    -----------
    To inherit the `DataContainer` class, you should know the following function:

    _retrieve_data(self, file_names, denoise=False, b0_threshold=None)
        This reads the properties of the given path based on the filenames and denoises the image.

    For correct inheritance, call the constructor with the correct filenames and
    pass denoise and threshold values. Example for HCP:

    >>> paths = {'bvals':'bvals', 'bvecs':'bvecs', 'img':'data.nii.gz',
                 't1':'T1w_acpc_dc_restore_1.25.nii.gz', 'mask':'nodif_brain_mask.nii.gz'}
    >>> DataContainer.__init__(self, path, paths, denoise=denoise, b0_threshold=b0_threshold)

    Then, your data is automatically correctly loaded and the other functions are working as well.
    """

    def __init__(self, path, file_names, denoise=None, b0_threshold=None):
        """
        Parameters
        ----------
        path : str
            The path leading to the DWI-Data Folder.
        file_names : dict
            A dictionary containg the file names for the specific values, e.g. 'img':'data.nii.gz'.
        denoise : bool, optional
            A boolean indicating wether the given data should be denoised,
            by default as in configuration file.
        b0_threshold : float, optional
            A single value indicating the b0 threshold used for b0 calculation,
            by default as in configuration file.
        """
        if denoise is None:
            denoise = Config.get_config().getboolean("data", "denoise", fallback="no")
        if b0_threshold is None:
            b0_threshold = Config.get_config().getfloat("data", "b0-threshold", fallback="10")
        self.options = SimpleNamespace()
        self.options.denoised = denoise
        self.options.cropped = False
        self.options.normalized = False
        self.options.b0_threshold = b0_threshold
        self.path = path.rstrip(os.path.sep)
        self.data = self._retrieve_data(file_names, denoise=denoise, b0_threshold=b0_threshold)
        self.id = ("DataContainer" + self.path.replace(os.path.sep, "-") + "-"
                   "b0thr-" + str(b0_threshold))
        if self.options.denoised:
            self.id = self.id + "-denoised"

    def _retrieve_data(self, file_names, denoise=False, b0_threshold=10):
        """
        Reads data from specific files and returns them as object.

        This functions reads the filenames of the DWI image and loads/parses them accordingly.
        Also, it denoises them, if specified and generates a b0 image.

        The `file_names` param should be a dict with the following keys:
        `['bvals', 'bvecs', 'img', 't1', 'mask']`

        Parameters
        ----------
        file_names : dict
            The filenames, or relative paths from `self.path`.
        denoise : bool, optional
            A boolean indicating wether the given data should be denoised, by default False
        b0_threshold : float, optional
            A single value indicating the b0 threshold used for b0 calculation, by default 10.0

        Returns
        -------
        SimpleNamespace
            An object holding all data as attributes, usable for further processing.

        Raises
        ------
        DataContainerNotLoadableError
            This error is thrown if one or multiple files cannot be foud.
        """
        data = SimpleNamespace()
        try:
            data.bvals, data.bvecs = read_bvals_bvecs(os.path.join(self.path, file_names['bvals']),
                                                      os.path.join(self.path, file_names['bvecs']))
            data.img = nb.load(os.path.join(self.path, file_names['img']))
            data.t1 = nb.load(os.path.join(self.path, file_names['t1'])).get_data()
        except FileNotFoundError as error:
            raise DataContainerNotLoadableError(self.path, error.filename) from None

        data.gtab = gradient_table(bvals=data.bvals, bvecs=data.bvecs)
        data.dwi = data.img.get_data().astype("float32")
        data.aff = data.img.affine
        data.fa = None

        if denoise:
            sigma = pca_noise_estimate(data.dwi, data.gtab, correct_bias=True,
                                       smooth=Config.get_config().getint("denoise", "smooth",
                                                                         fallback="3"))
            data.dwi = localpca(data.dwi, sigma=sigma,
                                patch_radius=Config.get_config().getint("denoise", "pathRadius",
                                                                        fallback="2"))
        if 'mask' in file_names:
            data.binarymask = nb.load(os.path.join(self.path, file_names['mask'])).get_data()
        else:
            _, data.binarymask = median_otsu(data.dwi[..., 0], 2, 1)

        data.b0 = data.dwi[..., data.bvals < b0_threshold].mean(axis=-1)

        return data

    def to_ijk(self, points):
        """
        Converts given RAS+ points to IJK in DataContainers Image Coordinates.

        The conversion happens using the affine of the DWI image.
        It should be noted that the dimension of the given points stays identical.

        Parameters
        ----------
        points : ndarray
            The points to convert.

        Returns
        -------
        ndarray
            The converted points.

        See also
        --------
        to_ras(points): the reverse method.
        """
        aff = np.linalg.inv(self.data.aff)
        return apply_affine(aff, points)

    def to_ras(self, points):
        """
        Converts given IJK points in DataContainers Coordinate System to RAS+.

        The conversion happens using the affine of the DWI image.
        It should be noted that the dimension of the given points stays identical.

        Parameters
        ----------
        points : ndarray
            The points to convert.

        Returns
        -------
        ndarray
            The converted points.

        See also
        --------
        to_ijk(points): the reverse method.
        """
        aff = self.data.aff
        return apply_affine(aff, points)

    def get_interpolated_dwi(self, points, ignore_outside_points=False):
        """
        Returns interpolated dwi for given RAS+ points.

        The shape of the input points will be retained for the return array,
        only the last dimension will be changed from 3 to the DWI-size accordingly.
        If `ignore_outside_points` is set to `True`,
        no error will be thrown for points outside of the image.

        Parameters
        ----------
        points : ndarray
            The array containing the points. Shape is matched in output.
        ignore_outside_points : bool, optional
            A boolean indicating wether points outside of the image should be ignored,
            by default False.

        Returns
        -------
        ndarray
            The DWI-Values interpolated for the given points.
            The input shape is matched aside of the last dimension.

        Raises
        ------
        PointOutsideOfDWIError
            This will be raised if some points are outside of the DWI image
            and ignore_outside_points isn't set to True.
        """
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

    def crop(self, b_value=None, max_deviation=None, ignore_already_cropped=False):
        """Crops the dataset based on B-value.

        This function crops the DWI-Image based on B-Value.
        Every value deviating more than `max_deviation` from the specified `b_value`
        will be irretrievably removed in `self`.

        Parameters
        ----------
        b_value : float, optional
            The b-value used for cropping, by default as in configuration.
        max_deviation : float, optional
            The maximum deviation allowed around given b-value, by default as in configuration.
        ignore_already_cropped : bool, optional
            If set to true, no `DWIAlreadyCroppedError` will be thrown even if applicable,
            by default `False`

        Returns
        -------
        DataContainer
            self after applying the crop.

        Raises
        ------
        DWIAlreadyCroppedError
            This error will be thrown to prevent multiple cropping which
            - because of the irretrievably - could lead to unexpected results.
            To use multiple cropping intentionally, set ignore_already_cropped to True.
        """
        if self.options.cropped and not ignore_already_cropped:
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
        """Normalize DWI Data based on b0 image.

        The weights are divided by their b0 value.

        Returns
        -------
        DataContainer
            self after applying the normalization.
        """
        if self.options.normalized:
            raise DWIAlreadyNormalizedError(self) from None
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
        self.options.normalized = True
        return self

    def get_fa(self):
        """Generates the FA Values for DataContainer.

        Normalization is required. It is advised to call the routine ahead of cropping!

        Returns
        -------
        Function
            Fractional anisotropy (FA) calculated from cached eigenvalues.
        """
        if not self.options.normalized:
            raise DWINotNormalizedError(self) from None
        if self.data.fa is None:
            dti_model = dti.TensorModel(self.data.gtab, fit_method='LS')
            dti_fit = dti_model.fit(self.data.dwi)
            self.data.fa = dti_fit.fa
        return self.data.fa

class HCPDataContainer(DataContainer):
    """
    The HCPDataContainer class is representing a single HCP Dataset.

    It contains basic functions to work with the data.
    The data itself is accessable in the `self.data` attribute.

    The `self.data` attribute contains the following
        - bvals: the B-values
        - bvecs: the B-vectors matching the bvals
        - img: the DWI-Image file
        - t1: the T1-File data
        - gtab: the calculated gradient table
        - dwi: the real DWI data
        - aff: the affine used for coordinate transformation
        - binarymask: a binarymask usable to separate brain from the rest
        - b0: the b0 image usable for normalization etc.

    Attributes
    ----------
    options: SimpleNamespace
        The configuration of the current DWI Data.
    path: str
        The path of the loaded DWI-Data.
    data: SimpleNamespace
        The dwi data, referenced in the SimpleNamespace's attributes.
    id: str
        An identifier of the current DWI-data including its preprocessing.
    hcp_id: int
        The HCP ID from which the data was retrieved.

    Methods
    -------
    to_ijk(points)
        Returns conversion of given RAS+ points into IJK format for DWI-File.
    to_ras(points)
        Returns conversion of given IJK points for DWI-File into RAS+ format.
    get_interpolated_dwi(points, ignore_outside_points=False)
        Returns 3D-interpolated DWI-Image values at the given RAS+ points.
    crop(b_value=None, max_deviation=None)
        Crops DWI-Data to given b_value and deviation. If param equals `None`,
        the values specified in the configuration file are used.
        Returns `self`.
    normalize()
        Normalizes the DWI-Image based on b0-Image.
        Returns `self`.
    get_fa()
        Generates the FA based on eigenvalues and returns them.
    """

    def __init__(self, hcpid, denoise=None, b0_threshold=None):
        """
        Parameters
        ----------
        hcpid : int
            The id of the HCP Dataset to load.
        denoise : bool, optional
            A boolean indicating wether the given data should be denoised,
            by default as in configuration file.
        b0_threshold : float, optional
            A single value indicating the b0 threshold used for b0 calculation,
            by default as in configuration file.
        """
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
    """
    The ISMRMDataContainer class is representing the artificial generated ISMRM Data.

    It contains basic functions to work with the data.
    The data itself is accessable in the `self.data` attribute.

    The `self.data` attribute contains the following
        - bvals: the B-values
        - bvecs: the B-vectors matching the bvals
        - img: the DWI-Image file
        - t1: the T1-File data
        - gtab: the calculated gradient table
        - dwi: the real DWI data
        - aff: the affine used for coordinate transformation
        - binarymask: a binarymask usable to separate brain from the rest
        - b0: the b0 image usable for normalization etc.

    Attributes
    ----------
    options: SimpleNamespace
        The configuration of the current DWI.
    path: str
        The path of the loaded DWI-Data.
    data: SimpleNamespace
        The dwi data, referenced in the SimpleNamespace's attributes.
    id: str
        An identifier of the current DWI-data including its preprocessing.

    Methods
    -------
    to_ijk(points)
        Returns conversion of given RAS+ points into IJK format for DWI-File.
    to_ras(points)
        Returns conversion of given IJK points for DWI-File into RAS+ format.
    get_interpolated_dwi(points, ignore_outside_points=False)
        Returns 3D-interpolated DWI-Image values at the given RAS+ points.
    crop(b_value=None, max_deviation=None)
        Crops DWI-Data to given b_value and deviation. If param equals `None`,
        the values specified in the configuration file are used.
        Returns `self`.
    normalize()
        Normalizes the DWI-Image based on b0-Image.
        Returns `self`.
    get_fa()
        Generates the FA based on eigenvalues and returns them.

    See Also
    --------
    src.tracker.ISMRMReferenceStreamlinesTracker: The streamlines matching this dataset.
    """
    def __init__(self, denoise=None, b0_threshold=None):
        """
        Parameters
        ----------
        denoise : bool, optional
            A boolean indicating wether the given data should be denoised,
            by default as in configuration file.
        b0_threshold : float, optional
            A single value indicating the b0 threshold used for b0 calculation,
            by default as in configuration file.
        """
        path = Config.get_config().get("data", "pathISMRM", fallback='data/ISMRM2015')
        paths = {'bvals':'Diffusion.bvals', 'bvecs':'Diffusion.bvecs',
                 'img':'Diffusion.nii.gz', 't1':'T1.nii.gz'}
        DataContainer.__init__(self, path, paths, denoise=denoise, b0_threshold=b0_threshold)

        self.id = "ISMRMDataContainer-b0thr-{b0}".format(b0=self.options.b0_threshold)
        if self.options.denoised:
            self.id = self.id + "-denoised"

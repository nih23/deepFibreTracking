"""
The dataset module is handling the datasets usable for training and testing.

Available subpackages
---------------------
processing
    Provides processing options for datasets.

Classes
-------
Error
    The base error class of all datasets.
WrongDatasetTypePassedError
    Error thrown if the passed Dataset has a wrong type.
FeatureShapesNotEqualError
    Error thrown if get_feature_shape is called on Dataset with multiple shapes
BaseDataset
    The base dataset every single dataset is based on. Inherits MovableData
IterableDataset
    Represents a dataset with fixed length and items.
ConcatenatedDataset
    A datasetclass used to concatenate multiple IterableDatasets
StreamlineDataset
    A Dataset consisting of streamlines and their DWI-Data.
"""
import torch
import numpy as np

from dipy.core.geometry import sphere_distance
from dipy.core.sphere import Sphere
from dipy.data import get_sphere

from src.data import MovableData
from types import SimpleNamespace
from src.config import Config
from src.util import get_reference_orientation, rotation_from_vectors, get_grid

class Error(Exception):
    """
    Base class for Dataset exceptions.

    Every Error happening from code of this class will inherit this one.
    The single parameter `msg` represents the error representing message.

    This class can be used to filter the exceptions for data exceptions.

    Attributes
    ----------
    message: str
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

    __str__ = __repr__

class WrongDatasetTypePassedError(Error):
    """Error thrown if `ConcatenatedDataset` retrieves wrong datasets.

    There are two cases in which this error will occur:
    1. The datasets you passed aren't exclusively IterableDatasets
    2. The specification of your datasets doesn't match and the option
        `ignore_data_specification` isn't set to True

    Attributes
    ----------
    caller: ConcatenatedDataset
        The ConcatenatedDataset raising the error.
    dataset: BaseDataset
        The dataset causig the error.
    msg: str
        The error message.
    """

    def __init__(self, concat, dataset, message):
        """
        Parameters
        ----------
        concat : ConcatenatedDataset
            The dataset raising the error.
        dataset : BaseDataset
            The dataset responsible for the error.
        message : str
            Your specific error message.
        """
        self.caller = concat
        self.dataset = dataset
        Error.__init__(self, msg=message)

class FeatureShapesNotEqualError(Error):
    """Error thrown if FeatureShapes of `ConcatenatedDataset` are not equal, but requested.

    This error only occurs, if `ConcatenatedDataset().get_feature_shape` is called,
    and the Datasets in the ConcatenatedDataset doesn't have equal feature shapes.

    Attributes
    ----------
    index: int
        The index of the BaseDataset responsible for the error.
    shape1: tuple
        The shape of the reference dataset.
    shape2: tuple
        The shape of the different dataset.
    msg: str
        The error message.
    """
    def __init__(self, index, s1, s2):
        """
        Parameters
        ----------
        index : int
            The index of the BaseDataset responsible for the error.
        s1 : tuple
            The shape of the reference dataset.
        s2 : tuple
            The shape of the dataset causing the error.
        """
        self.shape1 = s1
        self.shape2 = s2
        self.index = index
        Error.__init__(self, msg=("The shape of the dataset {idx} ({s2}) "
                                  "is not equal to the base shape of the reference dataset 0 ({s1})"
                                  ).format(idx=index, s2=s2, s1=s1))

class BaseDataset(MovableData):
    """The base class for Datasets in this library.

    It extends `MovableData`.
    
    Attributes
    ----------
    device: torch.device, optional
        The device the movable data currently is located on.
    data_container: DataContainer
        The DataContainer the dataset is based on
    id: str
        An ID representing this Dataset. This is not unique to any instance, but it consists of parameters and used dataset. 

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
    See `MovableData` for details.

    """
    def __init__(self, data_container, device=None):
        """
        Parameters
        ----------
        data_container: DataContainer
            The DataContainer the dataset uses
        device : torch.device, optional
            The device which the `MovableData` should be moved to on load, by default cpu.
        """
        MovableData.__init__(self, device=device)
        self.data_container = data_container
        self.id = str(self.__class__.__name__)
        if data_container is not None:
            self.id = self.id + "[" + str(data_container.id) + "]"


class IterableDataset(BaseDataset, torch.utils.data.Dataset):
    """The IterableDataset is the parent class of all maptype datasets.

    This class also implements torch.util.data.Dataset!
    Look up its attributes and methods too.

    It's childs should implement __len__ and __getitem__.
    Use this to check wether you are able to iterate over an unknown dataset.

    Attributes
    ----------
    device: torch.device, optional
        The device the movable data currently is located on.
    data_container: DataContainer
        The DataContainer the dataset is based on
    id: str
        An ID representing this Dataset. This is not unique to any instance, but it consists of parameters and used dataset. 

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
    See `MovableData` for details.
    """
    def __init__(self, data_container, device=None):
        """
        Parameters
        ----------
        data_container: DataContainer
            The DataContainer the dataset uses
        device : torch.device, optional
            The device which the `MovableData` should be moved to on load, by default cpu.
        """
        BaseDataset.__init__(self, data_container, device=device)
        torch.utils.data.Dataset.__init__(self)

    def __len__(self):
        if type(self) is IterableDataset:
            raise NotImplementedError() from None

    def __getitem__(self, index):
        if type(self) is IterableDataset:
            raise NotImplementedError() from None


class ConcatenatedDataset(IterableDataset):
    """A ConcatenatedDataset is an IterableDataset which is able to pack multiple IterableDatasets into a single one.
    
    This is pratical for combining the data of multiple DataContainers into a single dataset for training.
    It is advised to use the same type of data, but is not not necessary for using it. 
    To prevent data_specification checking, pass `ignore_data_specification=True` to the constructor.
    
    Attributes
    ----------
    device: torch.device, optional
        The device the movable data currently is located on.
    datasets: list
        A list of the datasets in this ConcatenatedDataset
    id: str
        An ID representing this Dataset. This is not unique to any instance, but it consists of parameters and used dataset.
    data_specification: str
        An string representing the type of Dataset this Concatenated dataset holds. If `ignore_data_specification=True`,
        it holds the specification of the first dataset.
    options: SimpleNamespace
        An Namespace holding all configuration options of this dataset.

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
    get_feature_shapes()
        Returns the feature shape of the given dataset. Useful for initializing NNs.

    Inheritance
    -----------
    See `MovableData` for details.
    """
    def __init__(self, datasets, device=None, ignore_data_specification=False):
        """
        Parameters
        ----------
        datasets: list
            A list of Datasets to use
        device : torch.device, optional
            The device which the `MovableData` should be moved to on load, by default cpu.
        ignore_data_specification : boolean, optional
            A boolean indicating wether the check of same dataset types should be ignored, default False.

        Raises
        ------
        WrongDatasetTypePassedError:
            Not all Datasets have the same type. To prevent this error from being thrown, pass `ignore_data_specification=True`
        """
        IterableDataset.__init__(self, None, device=device)
        self.id = self.id + "["
        self.__lens = [0]
        self.data_specification = datasets[0].data_specification
        for index, ds in enumerate(datasets):
            if not isinstance(ds, IterableDataset):
                raise WrongDatasetTypePassedError(self, ds,
                                                  ("Dataset {} doesn't inherit IterableDataset. "
                                                   "It is {} ").format(index, type(ds))
                                                 ) from None
            if ds.data_specification != self.data_specification and not ignore_data_specification:
                raise WrongDatasetTypePassedError(self, ds,
                                                  "Dataset {} doesn't match in DataSpecification."
                                                  .format(index)) from None
            ds.to(self.device)
            self.id = self.id + ds.id + ", "
            self.__lens.append(len(ds) + self.__lens[-1])
        self.id = self.id[:-2] + "]"
        self.datasets = datasets
        self.options = SimpleNamespace()
        self.options.ignore_data_specification = ignore_data_specification

    def __len__(self):
        return self.__lens[-1]
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('index {i} out of bounds for ConcatenatedDataset with length {l}.'
                             .format(i=index, l=len(self))) from None
        i = 0
        while self.__lens[i+1] <= index:
            i = i + 1

        return self.datasets[i][index - self.__lens[i]]

    def get_feature_shapes(self):
        """Returns the feature shapes as tuples (in, out).
        
        Raises
        ------
        FeatureShapesNotEqualError:
            Raised if this function is called on datasets with varying shapes.
        """
        # assert that each dataset has same dataset shape
        (inp, out) = self.datasets[0].get_feature_shapes()
        for i in range(1, len(self.datasets)):
            (inp2, out2) = self.datasets[i].get_feature_shapes()
            if (not torch.all(torch.tensor(inp).eq(torch.tensor(inp2))) or 
                    not torch.all(torch.tensor(out).eq(torch.tensor(out2)))):
                raise FeatureShapesNotEqualError(i, (inp, out), (inp2, out2))
        return (inp, out)

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
        ConcatenatedDataset
            The object moved to specified device
        """
        for dataset in self.datasets:
            dataset.cuda(device=device, non_blocking=non_blocking,
                         memory_format=memory_format)
            self.device = dataset.device
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
        ConcatenatedDataset
            The object moved to specified device
        """
        for dataset in self.datasets:
            dataset.cpu(memory_format=memory_format)
            self.device = dataset.device
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
        ConcatenatedDataset
            The object moved to specified device
        """
        for dataset in self.datasets:
            dataset.to(*args, **kwargs)
            self.device = dataset.device
        return self

class StreamlineDataset(IterableDataset):
    """Represents a dataset of streamlines. 
    
    Every item of the dataset is a tuple (input, output) with the dimensions 
    Streamline-Length x gridX x grid Y x gridZ x DWI-Data-Length for the input.

    You can pass this dataset to a dataloader. If `append_reverse` is True, 
    the inverse streamlines will be part of the dataset too. 

    The data will be generated and interpolated in runtime, 
    however this dataset is capable of caching the generated streamlines.
    Use this option only if you have enough (V)RAM. 

    Attributes
    ----------
    device: torch.device, optional
        The device the movable data currently is located on.
    data_container: DataContainer
        The DataContainer the dataset is based on
    streamlines: list
        A list containing all the streamlines in RAS+
    options: SimpleNamespace
        A namespace containing all specified options.
    cache: list, optional
        If online caching is active, the cache.
    id: str
        An ID representing this Dataset. This is not unique to any instance, but it consists of parameters and used dataset. 

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
    See `MovableData` for details regarding the Tensor device allocation. These functions are overwritten here.

    Also `_calculate_item(index)` generates an item and `_get_streamline(index)` retrieves the streamline in RAS+.

    """
    def __init__(self, tracker, data_container, processing,
                 device=None, append_reverse=None, ram_caching=None):
        """
        Parameters
        ----------
        tracker: Tracker
            The tracker which retrieved the streamlines.
        data_container: DataContainer
            The DataContainer matching the streamlines.
        processing: Processing
            A processing method to use with the data container
        device : torch.device, optional
            The device which the `MovableData` should be moved to on load, by default cpu.
        append_reverse: boolean, optional
            A boolean indicating wether the reversed streamlines should be appended to the Dataset.
        ram_caching : boolean, optional
            A boolean indicating wether the object should cache all generated streamlines.

        """
        IterableDataset.__init__(self, data_container, device=device)
        self.streamlines = tracker.get_streamlines()
        self.id = self.id + "-{}-(".format(processing.id) + tracker.id + ")"
        config = Config.get_config()
        if append_reverse is None:
            append_reverse = config.getboolean("DatasetOptions", "appendReverseStreamlines",
                                               fallback="yes")
        if ram_caching is None:
            ram_caching = config.getboolean("DatasetOptions", "ramCaching",
                                               fallback="yes")
        self.options = SimpleNamespace()
        self.options.append_reverse = append_reverse
        self.options.ram_caching = ram_caching
        self.options.processing = processing
        if ram_caching:
            self.cache = [None] * len(self)
        self.feature_shapes = None

    def __len__(self):
        if self.options.append_reverse:
            return 2*len(self.streamlines)
        return len(self.streamlines)

    def __getitem__(self, index):
        if self.options.ram_caching and self.cache[index] is not None:
            return self.cache[index]
        (inp, output) = self._calculate_item(index)
        inp = torch.from_numpy(inp).float().to(self.device)
        output = torch.from_numpy(output).float().to(self.device)

        if self.options.ram_caching:
            self.cache[index] = (inp, output)
            return self.cache[index]
        else:
            return (inp, output)

    def _calculate_item(self, index):
        """Calculates the input and output for given streamline identified by the index.

        Parameters
        ----------
        index, int:
            The index of the streamline.

        Returns
        -------
        object:
            The item calculated.
        """
        streamline = self._get_streamline(index)
        return self.options.processing.calculate_streamline(self.data_container, streamline)

    def _get_streamline(self, index):
        """Returns the requested streamline. 

        ! Use this function instead of the self.streamlines direct access because of the reversed streamlines.

        Parameters
        ----------
        index, int:
            The index of the streamline.

        Returns
        -------
        object:
            The streamline.
        """
        reverse = False
        if self.options.append_reverse and index >= len(self.streamlines):
            reverse = True
            index = index - len(self.streamlines)
        if reverse:
            streamline = self.streamlines[index][::-1]
        else:
            streamline = self.streamlines[index]
        return streamline


    def get_feature_shapes(self):
        """Returns the feature shapes as tuples (in, out).
        
        Raises
        ------
        FeatureShapesNotEqualError:
            Raised if this function is called on datasets with varying shapes.
        """
        if self.feature_shapes is None:
            dwi, next_dir = self[0]
            # assert that every type of data processing maintains same shape
            # and that every element has same shape
            input_shape = torch.tensor(dwi.shape)
            input_shape[0] = 1

            output_shape = torch.tensor(next_dir.shape)
            output_shape[0] = 1
            self.feature_shapes = (torch.prod(input_shape).item(), torch.prod(output_shape).item())
        return self.feature_shapes

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
        StreamlineDataset
            The object moved to specified device
        """
        if not self.options.ram_caching:
            return
        for index, el in enumerate(self.cache):
            if el is None:
                continue
            dwi, next_dir = el
            dwi = dwi.cuda(device=device, non_blocking=non_blocking,
                           memory_format=memory_format)
            next_dir = next_dir.cuda(device=device, non_blocking=non_blocking,
                                     memory_format=memory_format)
            self.cache[index] = (dwi, next_dir)
            if self.device == dwi.device:  # move is unnecessary
                return
        self.device = dwi.device
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
        StreamlineDataset
            The object moved to specified device
        """
        if not self.options.ram_caching:
            return
        for index, el in enumerate(self.cache):
            if el is None:
                continue
            dwi, next_dir = el
            dwi = dwi.cpu(memory_format=memory_format)
            next_dir = next_dir.cpu(memory_format=memory_format)
            self.cache[index] = (dwi, next_dir)
            if self.device == dwi.device:  # move is unnecessary
                return
        self.device = dwi.device
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
        ConcatenatedDataset
            The object moved to specified device
        """
        if not self.options.ram_caching:
            return
        for index, el in enumerate(self.cache):
            if el is None:
                continue
            dwi, next_dir = el
            dwi = dwi.to(*args, **kwargs)
            next_dir = next_dir.to(*args, **kwargs)
            self.cache[index] = (dwi, next_dir)
            if self.device == dwi.device: # move is unnecessary
                return
        self.device = dwi.device
        return self

class SingleDirectionsDataset(IterableDataset):
    """This Dataset is equivalent to StreamlineDataset, but more applicable for non-recurrent networks. 

    ! This element only support unrotated data !

    Every item just consists of single DWI (grid) values instead of a whole streamline, therefore
    one is able to work with constant batch sizes. The streamlines are sometimes split up over multiple 
    batches, if you combine it with an DataLoader.

    Every item of the dataset is a tuple (input, output) with the dimensions 
    gridX x grid Y x gridZ x DWI-Data-Length for the input.

    The data will be generated and interpolated in runtime, 
    however this dataset is capable of caching the generated streamlines.
    Use this option only if you have enough (V)RAM. 

    Attributes
    ----------
    device: torch.device, optional
        The device the movable data currently is located on.
    data_container: DataContainer
        The DataContainer the dataset is based on
    streamlines: list
        A list containing all the streamlines in RAS+
    size: int,
        The size of the Dataset
    calc_data: Tensor
        Preprocessed Data for generating the single elements live.
    options: SimpleNamespace
        A namespace containing all specified options.
    cache: list, optional
        If online caching is active, the cache.
    id: str
        An ID representing this Dataset. This is not unique to any instance, but it consists of parameters and used dataset. 

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
    See `MovableData` for details regarding the Tensor device allocation. These functions are overwritten here.

    Also `_calculate_item(index)` generates an item and `_get_streamline(index)` retrieves the streamline in RAS+.
    """
    def __init__(self, tracker, data_container, processing,
                 device=None, append_reverse=None, ram_caching=None):
        """

        ! Only unrotated data !
        Parameters
        ----------
        tracker: Tracker
            The tracker which retrieved the streamlines.
        data_container: DataContainer
            The DataContainer matching the streamlines.
        processing: Processing
            A processing method to use with the data container
        device : torch.device, optional
            The device which the `MovableData` should be moved to on load, by default cpu.
        append_reverse: boolean, optional
            A boolean indicating wether the reversed streamlines should be appended to the Dataset.
        ram_caching : boolean, optional
            A boolean indicating wether the object should cache all generated streamlines.
        """
        IterableDataset.__init__(self, data_container, device=device)

        #!! TODO Discuss ->  asserting every streamline is len 3 or longer
        append_end = False
        append_start = not processing.options.rotate
        #!!

        self.streamlines = tracker.get_streamlines()

        self.size = 0
        for streamline in self.streamlines:
            assert len(streamline) > 2
            self.size += len(streamline)

        if not append_end:
            self.size -= len(streamlines)
        if not append_start:
            self.size -= len(streamlines)


        self.id = self.id + "-{}-(".format(processing.id) + tracker.id + ")"
        config = Config.get_config()
        if append_reverse is None:
            append_reverse = config.getboolean("DatasetOptions", "appendReverseStreamlines",
                                               fallback="yes")
        if ram_caching is None:
            ram_caching = config.getboolean("DatasetOptions", "ramCaching",
                                               fallback="yes")
        self.options = SimpleNamespace
        self.options.append_reverse = append_reverse
        self.options.ram_caching = ram_caching
        self.options.processing = processing

        self.calc_data = np.zeros((self.size, 2))
        idx = 0
        _START_OFFSET = 0 if append_start else 1
        _END_OFFSET = 0 if append_end else 1
        for i, streamline in enumerate(self.streamlines):
            sl_len = len(streamline) - _START_OFFSET - _END_OFFSET
            self.calc_data[idx:(idx+sl_len)][0] = i
            self.calc_data[idx:(idx+sl_len)][1] = np.arange(_START_OFFSET, sl_len)
            idx += sl_len

        assert idx == self.size

        if ram_caching:
            self.cache = [] * len(self)
        self.feature_shapes = None

    def __len__(self):
        if self.options.append_reverse:
            return 2*self.size
        return self.size

    def get_feature_shapes(self):
        """Returns the feature shapes as tuples (in, out).
        
        Raises
        ------
        FeatureShapesNotEqualError:
            Raised if this function is called on datasets with varying shapes.
        """
        if self.feature_shapes is None:
            dwi, next_dir = self[0]
            # assert that every type of data processing maintains same shape
            # and that every element has same shape
            input_shape = torch.tensor(dwi.shape)
            input_shape[0] = 1

            output_shape = torch.tensor(next_dir.shape)
            output_shape[0] = 1
            self.feature_shapes = (torch.prod(input_shape).item(), torch.prod(output_shape).item())
        return self.feature_shapes
    def __getitem__(self, index):
        if self.options.ram_caching and self.cache[index] is not None:
            return self.cache[index]
        (inp, output) = self._calculate_item(index)
        inp = torch.from_numpy(inp).float().to(self.device)
        output = torch.from_numpy(output).float().to(self.device)

        if self.options.ram_caching:
            self.cache[index] = (inp, output)
            return self.cache[index]
        else:
            return (inp, output)

    def _calculate_item(self, index):
        """Calculates the input and output for given streamline identified by the index.

        Parameters
        ----------
        index, int:
            The index of the streamline.

        Returns
        -------
        object:
            The item calculated.
        """
        is_reverse = False
        if index >= self.size:
            is_reverse = True
            index = index - self.size
        (idx1, idx2) = self.calc_data[index]
        sl = self.streamlines[idx1]
        if not is_reverse:
            previous_sl = sl[:(idx2+1)]
            next_dir = sl[min(idx2+1, len(sl) - 1)] - sl[idx2] # is zero vector if empty
        else:
            previous_sl = sl[idx2:][::-1]
            next_dir = sl[max(idx2-1, 0)] - sl[idx2] # is zero vector if empty
        return self.options.processing.calculate_item(self.data_container, previous_sl, next_dir)


"""
The dataset module is handling the datasets usable for training and testing.

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

from src.data import MovableData, Object
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
    """The base class for Datasets in this library"""
    def __init__(self, data_container, device=None):
        MovableData.__init__(self, device=device)
        self.data_container = data_container
        self.id = str(self.__class__.__name__)
        if data_container is not None:
            self.id = self.id + "[" + str(data_container.id) + "]"


class IterableDataset(BaseDataset, torch.utils.data.Dataset):
    """Any map type dataset, implementing __len__ and __getitem__"""
    def __init__(self, data_container, device=None):
        BaseDataset.__init__(self, data_container, device=device)
        torch.utils.data.Dataset.__init__(self)

    def __len__(self):
        if type(self) is IterableDataset:
            raise NotImplementedError() from None

    def __getitem__(self, index):
        if type(self) is IterableDataset:
            raise NotImplementedError() from None


class ConcatenatedDataset(IterableDataset):
    """A class usable to concatenate multiple datasets.
    Same type is not necessary, but recommended for practical use.
    """
    def __init__(self, datasets, device=None, ignore_data_specification=False):
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
        self.options = Object()
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
        """Return feature shapes"""
        # assert that each dataset has same dataset shape
        (inp, out) = self.datasets[0].get_feature_shapes()
        for i in range(1, len(self.datasets)):
            (inp2, out2) = self.datasets[i].get_feature_shapes()
            if (not torch.all(torch.tensor(inp).eq(torch.tensor(inp2))) or 
                    not torch.all(torch.tensor(out).eq(torch.tensor(out2)))):
                raise FeatureShapesNotEqualError(i, (inp, out), (inp2, out2))
        return (inp, out)

    def cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        """Moves all Tensors to specified CUDA device"""
        for dataset in self.datasets:
            dataset.cuda(device=device, non_blocking=non_blocking,
                         memory_format=memory_format)
            self.device = dataset.device
        return self

    def cpu(self, memory_format=torch.preserve_format):
        """Moves all Tensors to CPU"""
        for dataset in self.datasets:
            dataset.cpu(memory_format=memory_format)
            self.device = dataset.device
        return self

    def to(self, *args, **kwargs):
        """Moves all tensors to specified device"""
        for dataset in self.datasets:
            dataset.to(*args, **kwargs)
            self.device = dataset.device
        return self

class StreamlineDataset(IterableDataset):
    """Represents a single dataset made of streamlines.
    In current implementation without caching"""
    def __init__(self, tracker, data_container, processing,
                 device=None, append_reverse=None, online_caching=None):
        IterableDataset.__init__(self, data_container, device=device)
        self.streamlines = tracker.get_streamlines()
        self.id = self.id + "-{}-(".format(processing.id) + tracker.id + ")"
        config = Config.get_config()
        if append_reverse is None:
            append_reverse = config.getboolean("DatasetOptions", "appendReverseStreamlines",
                                               fallback="yes")
        if online_caching is None:
            online_caching = config.getboolean("DatasetOptions", "onlineCaching",
                                               fallback="yes")
        self.options = Object()
        self.options.append_reverse = append_reverse
        self.options.online_caching = online_caching
        self.options.processing = processing
        if online_caching:
            self.cache = [None] * len(self)
        self.feature_shapes = None

    def __len__(self):
        if self.options.append_reverse:
            return 2*len(self.streamlines)
        return len(self.streamlines)

    def __getitem__(self, index):
        if self.options.online_caching and self.cache[index] is not None:
            return self.cache[index]
        (inp, output) = self._calculate_item(index)
        inp = torch.from_numpy(inp).float().to(self.device)
        output = torch.from_numpy(output).float().to(self.device)

        if self.options.online_caching:
            self.cache[index] = (inp, output)
            return self.cache[index]
        else:
            return (inp, output)

    def _calculate_item(self, index):
        streamline = self._get_streamline(index)
        return self.options.processing.calculate_streamline(self.data_container, streamline)

    def _get_streamline(self, index):
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
        """Retrieve feature shape of in and output for neural network"""
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
        """Moves all Tensors to specified CUDA device"""
        if not self.options.online_caching:
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
        """Moves all Tensors to specified CUDA device"""
        if not self.options.online_caching:
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
        """Moves all Tensors to specified device"""
        if not self.options.online_caching:
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
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
import json
import os
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

    This means that the datasets you passed aren't exclusively IterableDatasets

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
    def __init__(self, data_container, device=None):
        BaseDataset.__init__(self, data_container, device=device)
        torch.utils.data.Dataset.__init__(self)

    def __len__(self):
        if type(self) is IterableDataset:
            raise NotImplementedError() from None

    def __getitem__(self, index):
        if type(self) is IterableDataset:
            raise NotImplementedError() from None


class SaveableDataset(IterableDataset):
    def __init__(self, data_container, device=None):
        IterableDataset.__init__(self,data_container, device=device)
    
    def _get_variable_elements_data(self):
        lengths = np.zeros(len(self), dtype=int)
        for i, (inp, out) in enumerate(self):
            assert len(inp) == len(out)
            lengths[i] = len(inp)
        return lengths, inp.shape[1:], out.shape[1:]

    def saveToPath(self, path):

        os.makedirs(path, exist_ok=True)
        lengths, in_shape, out_shape = self._get_variable_elements_data()
        print(lengths)
        data_length = int(np.sum(lengths))

        in_shape=tuple([data_length] + list(in_shape))
        out_shape=tuple([data_length] + list(out_shape))

        inp_memmap = np.memmap(os.path.join(path, 'input.npy'), dtype='float32', shape=in_shape, mode='w+')
        out_memmap = np.memmap(os.path.join(path, 'output.npy'), dtype='float32', shape=out_shape, mode='w+')
        
        idx = 0
        assert (len(self) == len(lengths))
        for i in range(len(self)):
            inp,out = self[i]
            print(i, ": ", inp.shape, " l " ,lengths[i],  " - ",lengths.shape)
            assert(len(inp) == lengths[i])
            inp_memmap[idx:(idx + lengths[i])] = inp.numpy()
            out_memmap[idx:(idx + lengths[i])] = out.numpy()
            idx = idx +  lengths[i] 
            print("{}/{}".format(i, len(lengths)), end="\r")
        np.save(os.path.join(path, 'lengths.npy'), lengths)
        with open(os.path.join(path, 'info.json'), 'w') as infofile:
            json.dump({"id": self.id, "input_shape":in_shape, "output_shape":out_shape}, infofile)


class LoadedDataset(IterableDataset):
    def __init__(self, path, device=None, passSingleElements=False):
        IterableDataset.__init__(self, None, device=device)
        self.path = path

        with open(os.path.join(self.path, 'info.json')) as infofile:
            info_data = json.load(infofile)
        self.id = info_data["id"] + "-loaded"

        inp_shape = tuple(info_data["input_shape"])
        out_shape = tuple(info_data["output_shape"])
        self.feature_shapes = np.prod(info_data["input_shape"][1:]), np.prod(info_data["output_shape"][1:])
        #self.id = "LoadedDataset-{}".format(path.replace(os.path.sep,"-"))
        if not passSingleElements:
            self.sl_lengths = np.load(os.path.join(self.path, 'lengths.npy'))
        else:
            self.sl_lengths = np.ones((inp_shape[0]))
        self.sl_start_indices = np.append(0, np.cumsum(self.sl_lengths))

        self.inp_memmap = np.memmap(os.path.join(self.path, 'input.npy'), dtype='float32', shape=inp_shape, mode='r')
        self.out_memmap = np.memmap(os.path.join(self.path, 'output.npy'), dtype='float32', shape=out_shape, mode='r')
    
    def __len__(self):
        return len(self.sl_lengths)

    def __getitem__(self, index):
        inp = torch.from_numpy(self.inp_memmap[self.sl_start_indices[index]:self.sl_start_indices[index+1]]).to(self.device)
        out = torch.from_numpy(self.out_memmap[self.sl_start_indices[index]:self.sl_start_indices[index+1]]).to(self.device)
        return (inp, out)

    def get_feature_shapes(self):
        return self.feature_shapes

class ConcatenatedDataset(SaveableDataset):
    def __init__(self, datasets, device=None):
        IterableDataset.__init__(self, None, device=device)
        self.id = self.id + "["
        self.__lens = [0]
        for index, ds in enumerate(datasets):
            if not isinstance(ds, IterableDataset):
                raise WrongDatasetTypePassedError(self, ds,
                                                  ("Dataset {} doesn't inherit IterableDataset. "
                                                   "It is {} ").format(index, type(ds))
                                                 ) from None
            ds.to(self.device)
            self.id = self.id + ds.id + ", "
            self.__lens.append(len(ds) + self.__lens[-1])
        self.id = self.id[:-2] + "]"
        self.datasets = datasets
        self.options = SimpleNamespace()

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
        # assert that each dataset has same dataset shape
        (inp, out) = self.datasets[0].get_feature_shapes()
        for i in range(1, len(self.datasets)):
            (inp2, out2) = self.datasets[i].get_feature_shapes()
            if (not torch.all(torch.tensor(inp).eq(torch.tensor(inp2))) or 
                    not torch.all(torch.tensor(out).eq(torch.tensor(out2)))):
                raise FeatureShapesNotEqualError(i, (inp, out), (inp2, out2))
        return (inp, out)

    def cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        for dataset in self.datasets:
            dataset.cuda(device=device, non_blocking=non_blocking,
                         memory_format=memory_format)
            self.device = dataset.device
        return self

    def cpu(self, memory_format=torch.preserve_format):
        for dataset in self.datasets:
            dataset.cpu(memory_format=memory_format)
            self.device = dataset.device
        return self

    def to(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.to(*args, **kwargs)
            self.device = dataset.device
        return self

class StreamlineDataset(SaveableDataset):

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
        self.options = SimpleNamespace()
        self.options.append_reverse = append_reverse
        self.options.online_caching = online_caching
        self.options.processing = processing
        if online_caching:
            self.cache = [None] * len(self)
        self.feature_shapes = None
    
    def _get_variable_elements_data(self):
        lengths = np.zeros(len(self) , dtype=int)
        for i, sl in enumerate(self.streamlines):
            lengths[i] = len(sl)
        if self.options.append_reverse:
            lengths[len(self.streamlines):] = lengths[:len(self.streamlines)]
        (inp, out) = self[0]
        return lengths, inp.shape[1:], out.shape[1:]


    def __len__(self):
        if self.options.append_reverse:
            return 2*len(self.streamlines)
        return len(self.streamlines)

    def __getitem__(self, index):
        if self.options.online_caching and self.cache[index] is not None:
            return self.cache[index]
        (inp, output) = self._calculate_item(index)
        inp = torch.from_numpy(inp).to(device=self.device, dtype=torch.float32) # TODO work on dtypes
        output = torch.from_numpy(output).to(device=self.device, dtype=torch.float32)

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

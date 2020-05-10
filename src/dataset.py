"""Class responsible for handling datasets"""
import torch

from src.data import MovableData, Object
from src.config import Config

class Error(Exception):
    """Base class for Dataset exceptions."""

    def __init__(self, msg=''):
        self.message = msg
        Exception.__init__(self, msg)

    def __repr__(self):
        return self.message

    __str__ = __repr__

class WrongDatasetTypePassedError(Error):
    """Error thrown if get_device is called on CUDA tensor."""

    def __init__(self, concat, dataset, message):
        self.caller = concat
        self.dataset = dataset
        Error.__init__(self, msg=message)

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
    def __init__(self, streamlines, data_container, grid_dimension=(3,3,3)):
        self.streamlines = streamlines
        self.data_container = data_container
        if grid_dimension is None:
            grid_dimension = (Config.get_config().getint("GridOptions", "sizeX", fallback="3"),
                              Config.get_config().getint("GridOptions", "sizeY", fallback="3"),
                              Config.get_config().getint("GridOptions", "sizeZ", fallback="3"))
        self.options = Object()
        self.options.grid_dimension = grid_dimension

    def __get_grid(self, grid_dimension):
        pass

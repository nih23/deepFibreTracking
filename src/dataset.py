"""Class responsible for handling datasets"""
import torch
import numpy as np

from src.data import MovableData, Object
from src.config import Config
from src.util import get_reference_orientation, rotation_from_vectors

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
    def __init__(self, tracker, data_container, rotate=None, grid_dimension=None, grid_spacing=None,
                 device=None, append_reverse=None, online_caching=None):
        IterableDataset.__init__(self, data_container, device=device)
        self.streamlines = tracker.get_streamlines()
        self.id = self.id + "-(" + tracker.id + ")"
        config = Config.get_config()
        if grid_dimension is None:
            grid_dimension = np.array((config.getint("GridOptions", "sizeX", fallback="3"),
                                       config.getint("GridOptions", "sizeY", fallback="3"),
                                       config.getint("GridOptions", "sizeZ", fallback="3")))
        if isinstance(grid_dimension, tuple):
            grid_dimension = np.array(grid_dimension)
        if grid_spacing is None:
            grid_spacing = config.getfloat("GridOptions", "spacing", fallback="1.0")
        if append_reverse is None:
            append_reverse = config.getboolean("DatasetOptions", "appendReverseStreamlines",
                                               fallback="yes")
        if rotate is None:
            rotate = config.getboolean("DatasetOptions", "rotateDataset",
                                       fallback="yes")
        if online_caching is None:
            online_caching = config.getboolean("DatasetOptions", "onlineCaching",
                                               fallback="yes")
        self.options = Object()
        self.options.rotate = rotate
        self.options.append_reverse = append_reverse
        self.options.grid_dimension = grid_dimension
        self.options.grid_spacing = grid_spacing
        self.options.online_caching = online_caching
        self.data_specification = "raw-{x}x{y}x{z}-{sw}".format(x=grid_dimension[0],
                                                                y=grid_dimension[1],
                                                                z=grid_dimension[2],
                                                                sw=grid_spacing)
        if rotate:
            self.data_specification = self.data_specification + "-rotated"
        if online_caching:
            self.cache = [None] * len(self)
        self.grid = self._get_grid(grid_dimension) * grid_spacing

    def _get_grid(self, grid_dimension):
        (dx, dy, dz) = (grid_dimension - 1)/2
        return np.moveaxis(np.array(np.mgrid[-dx:dx+1, -dy:dy+1, -dz:dz+1]), 0, 3)

    def __len__(self):
        if self.options.append_reverse:
            return 2*len(self.streamlines)
        return len(self.streamlines)

    def __getitem__(self, index):
        if self.options.online_caching and self.cache[index] is not None:
            return self.cache[index]
        reverse = False
        if self.options.append_reverse and index >= len(self.streamlines):
            reverse = True
            index = index - len(self.streamlines)
        if reverse:
            streamline = self.streamlines[index][::-1]
        else:
            streamline = self.streamlines[index]
        next_dir, rot_matrix = self._get_next_direction(streamline, rotate=self.options.rotate)
        dwi, _ = self._get_dwi(streamline, rot_matrix=rot_matrix)
        dwi = torch.from_numpy(dwi).to(self.device)
        next_dir = torch.from_numpy(next_dir).to(self.device)
        if self.options.online_caching:
            self.cache[index] = (dwi, next_dir)
        return dwi, next_dir

    def _get_next_direction(self, streamline, rotate=False):
        next_dir = streamline[1:] - streamline[:-1]
        next_dir = np.concatenate((next_dir, np.array([[0, 0, 0]])))
        rot_matrix = None

        if rotate:
            reference = get_reference_orientation()
            rot_matrix = np.empty([len(next_dir), 3, 3])
            rot_matrix[0] = np.eye(3)
            for i in range(len(next_dir) - 1):
                rotation_from_vectors(rot_matrix[i + 1], reference, next_dir[i])
                next_dir[i] = rot_matrix[i].T @ next_dir[i]

        return next_dir, rot_matrix

    def _get_dwi(self, streamline, rot_matrix=None):
        points = self._get_grid_points(streamline, rot_matrix=rot_matrix)
        return self.data_container.get_interpolated_dwi(points), points

    def _get_grid_points(self, streamline, rot_matrix=None):
        grid = self.grid
        if rot_matrix is None:
            applied_grid = grid # grid is static
            # shape [R x A x S x 3]
        else:
            # grid is rotated for each streamline_point
            applied_grid = ((rot_matrix.repeat(grid.size/3, axis=0) @
                             grid[None,].repeat(len(streamline), axis=0).reshape(-1, 3, 1))
                            .reshape((-1, *grid.shape)))
            # shape [N x R x A x S x 3]

        points = streamline[:, None, None, None, :] + applied_grid
        return points

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

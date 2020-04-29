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
    Attributes:

        device
            A torch.device, representing the current device.
"""

import torch


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

    def cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        """Moves all Tensors to specified CUDA device"""
        for attribute, tensor in self._get_tensors().items():
            cuda_tensor = tensor.cuda(device=device, non_blocking=non_blocking,
                                      memory_format=memory_format)
            setattr(self, attribute, cuda_tensor)
            self.device = cuda_tensor.device
        return self

    def cpu(self, memory_format=torch.preserve_format):
        """Moves all Tensors to CPU"""
        for attribute, tensor in self._get_tensors().items():
            cpu_tensor = tensor.cpu(memory_format=memory_format)
            setattr(self, attribute, cpu_tensor)
        self.device = torch.device('cpu')
        return self

    def to(self, *args, **kwargs):
        """Moves all tensors to specified device"""
        for attribute, tensor in self._get_tensors().items():
            tensor = tensor.to(*args, **kwargs)
            setattr(self, attribute, tensor)
            self.device = tensor.device
        return self

    def get_device(self):
        """Returns CUDA device number. Throwns a DeviceNotRetrievableError if on CPU."""
        if self.device.type == "cpu":
            raise DeviceNotRetrievableError(self.device)
        return self.device.index

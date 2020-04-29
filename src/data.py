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
    device = None
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cpu")
        self.device = device

    def _get_tensors(self):
        tensors = {}
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
        return tensors

    def cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        for attribute, tensor in self._get_tensors().items():
            cuda_tensor = tensor.cuda(device=device, non_blocking=non_blocking,
                                      memory_format=memory_format)
            setattr(self, attribute, cuda_tensor)
            self.device = cuda_tensor.device

    def cpu(self, memory_format=torch.preserve_format):
        for attribute, tensor in self._get_tensors().items():
            cpu_tensor = tensor.cpu(memory_format=memory_format)
            setattr(self, attribute, cpu_tensor)
        self.device = torch.device('cpu')

    def to(self, *args, **kwargs):
        for attribute, tensor in self._get_tensors().items():
            tensor = tensor.to(*args, **kwargs)
            setattr(self, attribute, tensor)
            self.device = tensor.device

    def get_device(self):
        if self.device.type == "cpu":
            raise DeviceNotRetrievableError(self.device)
        return self.device.index

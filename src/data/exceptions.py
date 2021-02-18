class Error(Exception):
    """
    Base class for Data exceptions.

    Every Error happening from code of this class will inherit this one.
    The single parameter `msg` represents the error representing message.

    This class can be used to filter the exceptions for data exceptions.
    """

    def __init__(self, msg):
        """
        Parameters
        ----------
        msg : str
            The message describing the error
        """
        Exception.__init__(self, msg)


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
    path: str
        The path in which the software was unable to find the file.
    file: str
        The filename which couldn't be found in folder.
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

class DWIAlreadyCroppedError(Error):
    """
    Error thrown if the DWI data is cropped multiple times.

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

    You have to create a new DataContainer if you want to change the normalization

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
                       .format(no_points=len(points), id=data_container.id, aff=affected_points))
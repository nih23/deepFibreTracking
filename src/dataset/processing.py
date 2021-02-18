"""The processing submodule contains processing options for the raw streamline and DWI data.

Classes
-------
Processing
    The base class for all processing instructions
RegressionProcessing
    The basic processing, calculates direction vectors out of streamlines and interpolates DWI along a grid 
ClassificationProcessing
    Based on RegressionProcessing, however it reshapes the regression problem of the direction vector as a classification problem.
"""
from types import SimpleNamespace
import numpy as np
import torch
from dipy.core.geometry import sphere_distance
from dipy.core.sphere import Sphere
from dipy.data import get_sphere

from src.config import Config
from src.util import get_reference_orientation, rotation_from_vectors, get_grid, apply_rotation_matrix_to_grid, direction_to_classification, rotation_from_vectors_p

class Processing():
    """The basic Processing class.

    Every Processing should extend this function and implement the following:

    Methods
    -------
    calculate_streamline(data_container, streamline)
        Calculates the (input, output) tuple for a complete streamline
    calculate_item(data_container, sl, next_direction)
        Calculates the (input, output) tuple for a single last streamline point

    The methods can work together, but they do not have to. 
    The existence of both must be guaranteed to be able to use every dataset.
    """
    # TODO - Live Calculation for Tracker
    def calculate_streamline(self, data_container, streamline):
        """Calculates the (input, output) tuple for a whole streamline.

        Arguments
        ---------
        data_container : DataContainer
            The DataContainer the streamline is associated with
        streamline: Tensor
            The streamline the input and output data should be calculated for

        Raises
        ------
        NotImplementedError
            If the Processing subclass didn't overwrite the function.
        
        Returns
        -------
        tuple
            The (input, output) data for the requested item.

        """
        raise NotImplementedError
    def calculate_item(self, data_container, previous_sl, next_dir):
        """Calculates the (input, output) tuple for a single streamline point.

        Arguments
        ---------
        data_container : DataContainer
            The DataContainer the streamline is associated with
        point: Tensor
            The point the data should be calculated for in RAS*
        next_dir: Tensor, optional
            The next direction, you do not have to provide it if you only need the input part.
        Raises
        ------
        NotImplementedError
            If the Processing subclass didn't overwrite the function.

        Returns
        -------
        tuple
            The (input, output) data for the requested item.
        """
        raise NotImplementedError

class RegressionProcessing(Processing):
    """Provides a Processing option for regression training.

    There are many configuration options specified in the constructor.
    An instance of this class has to be passed onto a Dataset.

    Attributes
    ----------
    options: SimpleNamespace
        An object holding all configuration options of this dataset.
    grid: numpy.ndarray
        The grid, precalculated for this processing option
    id: str
        An ID representing this Dataset. This is not unique to any instance, but it consists of parameters and used dataset. 

    Methods
    -------
    calculate_streamline(data_container, streamline)
        Calculates the (input, output) tuple for a complete streamline
    calculate_item(data_container, point, next_direction)
        Calculates the (input, output) tuple for a single streamline point

    """
    def __init__(self, rotate=None, grid_dimension=None, grid_spacing=None, postprocessing=None, normalize=None, normalize_mean=None, normalize_std=None):
        """

        If the parameters are passed as none, the value from the config.ini is used.

        Parameters
        ----------
        rotate : bool, optional
            Indicates wether grid should be rotated along fiber, by default None
        grid_dimension : numpy.ndarray, optional
            Grid dimension (X,Y,Z) of the interpolation grid, by default None
        grid_spacing : float, optional
            Grid spacing, by default None
        postprocessing : data.postprocessing, optional
            The postprocessing to be done on the interpolated DWI, by default None
        normalize : bool, optional
            Indicates wether data should be normalized, by default None
        normalize_mean : numpy.ndarray, optional
            Give mean for normalization, by default None
        normalize_std : numpy.ndarray, optional
            Give std for normalization, by default None
        """
        config = Config.get_config()
        if grid_dimension is None:
            grid_dimension = np.array((config.getint("GridOptions", "sizeX", fallback="3"),
                                       config.getint("GridOptions", "sizeY", fallback="3"),
                                       config.getint("GridOptions", "sizeZ", fallback="3")))

        if isinstance(grid_dimension, tuple):
            grid_dimension = np.array(grid_dimension)

        

        if grid_spacing is None:
            grid_spacing = config.getfloat("GridOptions", "spacing", fallback="1.0")
        if rotate is None:
            rotate = config.getboolean("Processing", "rotateDataset",
                                       fallback="yes")
        if rotate and normalize is None:
            normalize = config.getboolean("Processing", "normalizeRotatedDataset",
                                          fallback="yes")
        else:
            normalize = False

        self.options = SimpleNamespace()

        if rotate and normalize:
            if normalize_mean is None:
                normalize_mean = np.array((config.getfloat("RotationNorm", "meanX",
                                                           fallback="9.8811e-01"),
                                           config.getfloat("RotationNorm", "meanY",
                                                           fallback="2.6814e-04"),
                                           config.getfloat("RotationNorm", "meanZ",
                                                           fallback="1.2876e-03")))
            if isinstance(normalize_mean, tuple):
                normalize_mean = np.array(normalize_mean)

            if normalize_std is None:
                normalize_std = np.array((config.getfloat("RotationNorm", "stdX",
                                                          fallback="0.0262"),
                                          config.getfloat("RotationNorm", "stdY",
                                                          fallback="0.1064"),
                                          config.getfloat("RotationNorm", "stdZ",
                                                          fallback="0.1078")))
            if isinstance(normalize_std, tuple):
                normalize_std = np.array(normalize_std)

            self.options.normalize_mean = normalize_mean
            self.options.normalize_std = normalize_std

        self.options.rotate = rotate
        self.options.normalize = normalize
        self.options.grid_dimension = grid_dimension
        self.options.grid_spacing = grid_spacing
        self.options.postprocessing = postprocessing
        self.grid = get_grid(grid_dimension) * grid_spacing

        self.id = "RegressionProcessing-r{}-grid{}x{}x{}-spacing{}-postprocessing-{}".format(rotate, *grid_dimension, grid_spacing, postprocessing.id)

    def calculate_item(self, data_container, previous_sl, next_dir):
        """Calculates the (input, output) tuple for the last streamline point.

        Arguments
        ---------
        data_container : DataContainer
            The DataContainer the streamline is associated with
        previous_sl: np.array
            The previous streamline point including the point the data should be calculated for in RAS*
        next_dir: Tensor
            The next direction, provide a null vector [0,0,0] if it is irrelevant.

        Returns
        -------
        tuple
            The (input, output) data for the requested item.
        """
        # create artificial next_dirs consisting of last and next dir for rot_mat calculation
        next_dirs = np.concatenate(((previous_sl[1:] - previous_sl[:-1])[-1:], next_dir[np.newaxis, ...])) 
        # TODO - normalize direction vectors
        next_dirs, rot_matrix = self._apply_rot_matrix(next_dirs)
        
        next_dir = next_dirs[-1]
        rot_matrix = None if rot_matrix is None else rot_matrix[np.newaxis, -1]
        dwi, _ = self._get_dwi(data_container, previous_sl[np.newaxis, -1], rot_matrix=rot_matrix)
        if self.options.postprocessing is not None:
            dwi = self.options.postprocessing(dwi, data_container.data.b0,
                                              data_container.data.bvecs,
                                              data_container.data.bvals)
        dwi = dwi.squeeze(axis=0)
        if self.options.normalize:
            next_dir = (next_dir - self.options.normalize_mean)/self.options.normalize_std
        return dwi, next_dir

    def calculate_streamline(self, data_container, streamline):
        """Calculates the (input, output) tuple for a whole streamline.

        Arguments
        ---------
        data_container : DataContainer
            The DataContainer the streamline is associated with
        streamline: Tensor
            The streamline the input and output data should be calculated for
        
        Returns
        -------
        tuple
            The (input, output) data for the requested item.

        """
        next_dir = self._get_next_direction(streamline)
        next_dir, rot_matrix = self._apply_rot_matrix(next_dir)
        dwi, _ = self._get_dwi(data_container, streamline, rot_matrix=rot_matrix, postprocessing=self.options.postprocessing)
        if self.options.postprocessing is not None:
            dwi = self.options.postprocessing(dwi, data_container.data.b0,
                                              data_container.data.bvecs,
                                              data_container.data.bvals)
        if self.options.normalize:
            next_dir = (next_dir - self.options.normalize_mean)/self.options.normalize_std
        return (dwi, next_dir)

    def _get_dwi(self, data_container, streamline, rot_matrix=None, postprocessing=None):
        points = self._get_grid_points(streamline, rot_matrix=rot_matrix)
        dwi = data_container.get_interpolated_dwi(points, postprocessing=postprocessing) 
        return dwi , points

    def _get_next_direction(self, streamline):
        next_dir = streamline[1:] - streamline[:-1]
        next_dir = next_dir / np.linalg.norm(next_dir, axis=1)[:, None]
        next_dir = np.concatenate((next_dir, np.array([[0, 0, 0]])))
        return next_dir

    def _apply_rot_matrix(self, next_dir):
        if not self.options.rotate:
            return next_dir, None
        reference = get_reference_orientation()
        rot_matrix = np.empty([len(next_dir), 3, 3])
        # rot_mat (N, 3, 3)
        # next dir (N, 3)
        rot_matrix[0] = np.eye(3)
        rotation_from_vectors_p(rot_matrix[1:, :, :], reference[None, :], next_dir[:-1])

        rot_next_dir = (rot_matrix.transpose((0,2,1))  @ next_dir[:, :, None]).squeeze(2)
        return rot_next_dir, rot_matrix
        

    def _get_grid_points(self, streamline, rot_matrix=None):
        grid = self.grid
        if rot_matrix is not None:
            grid = apply_rotation_matrix_to_grid(grid, rot_matrix)
            # shape [N x R x A x S x 3] or [R x A x S x 3]
        points = streamline[:, None, None, None, :] + grid
        return points


class ClassificationProcessing(RegressionProcessing):
    """Provides a Processing option for regression training.

    There are many configuration options specified in the constructor.
    An instance of this class has to be passed onto a Dataset.

    Attributes
    ----------
    options: SimpleNamespace
        An object holding all configuration options of this dataset.
    grid: numpy.ndarray
        The grid, precalculated for this processing option
    id: str
        An ID representing this Dataset. This is not unique to any instance, but it consists of parameters and used dataset. 

    Methods
    -------
    calculate_streamline(data_container, streamline)
        Calculates the (input, output) tuple for a complete streamline
    calculate_item(data_container, point, next_direction)
        Calculates the (input, output) tuple for a single streamline point
    """
    def __init__(self, rotate=None, grid_dimension=None, grid_spacing=None, postprocessing=None,
                 sphere=None):
        """

        If the parameters are passed as none, the value from the config.ini is used.

        Parameters
        ----------
        rotate : bool, optional
            Indicates wether grid should be rotated along fiber, by default None
        grid_dimension : numpy.ndarray, optional
            Grid dimension (X,Y,Z) of the interpolation grid, by default None
        grid_spacing : float, optional
            Grid spacing, by default None
        postprocessing : data.postprocessing, optional
            The postprocessing to be done on the interpolated DWI, by default None
        sphere : Sphere or str, optional
            The sphere to use for interpolation
        """

        RegressionProcessing.__init__(self, rotate=rotate, grid_dimension=grid_dimension,
                                      grid_spacing=grid_spacing, postprocessing=postprocessing,
                                      normalize=False)
        if sphere is None:
            sphere = Config.get_config().get("Processing", "classificationSphere",
                                             fallback="repulsion724")
        if isinstance(sphere, Sphere):
            rsphere = sphere
            sphere = "custom"
        else:
            rsphere = get_sphere(sphere)
        self.sphere = rsphere
        self.options.sphere = sphere
        self.id = ("ClassificationProcessing-r{}-sphere-{}-grid{}x{}x{}-spacing{}-postprocessing-{}"
                   .format(self.options.rotate, self.options.sphere, *self.options.grid_dimension, self.options.grid_spacing, self.options.postprocessing.id))

    def calculate_streamline(self, data_container, streamline):
        """Calculates the classification (input, output) tuple for a whole streamline.

        Arguments
        ---------
        data_container : DataContainer
            The DataContainer the streamline is associated with
        streamline: Tensor
            The streamline the input and output data should be calculated for
        
        Returns
        -------
        tuple
            The (input, output) data for the requested item.

        """
        dwi, next_dir = RegressionProcessing.calculate_streamline(self, data_container, streamline)
        classification_output = direction_to_classification(self.sphere, next_dir, include_stop=True, last_is_stop=True)
        return dwi, classification_output

    def calculate_item(self, data_container, previous_sl, next_dir):
        """Calculates the classification (input, output) tuple for the last streamline point.

        Arguments
        ---------
        data_container : DataContainer
            The DataContainer the streamline is associated with
        previous_sl: np.array
            The previous streamline point including the point the data should be calculated for in RAS*
        next_dir: Tensor
            The next direction, provide a null vector [0,0,0] if it is irrelevant.

        Returns
        -------
        tuple
            The (input, output) data for the requested item.
        """
        dwi, next_dir = RegressionProcessing.calculate_item(data_container, previous_sl, next_dir)
        classification_output = direction_to_classification(self.sphere, next_dir[None, ...], include_stop=True, last_is_stop=True).squeeze(axis=0)
        return dwi, classification_output
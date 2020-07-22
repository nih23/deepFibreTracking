from types import SimpleNamespace
import numpy as np
from dipy.core.geometry import sphere_distance
from dipy.core.sphere import Sphere
from dipy.data import get_sphere

from src.config import Config
from src.util import get_reference_orientation, rotation_from_vectors, get_grid

class Processing():
    # TODO - Live Calculation for Tracker
    def calculate_streamline(self, data_container, streamline):
        raise NotImplementedError

class RegressionProcessing(Processing):
    def __init__(self, rotate=None, grid_dimension=None, grid_spacing=None, postprocessing=None, normalize=None, normalize_mean=None, normalize_std=None):
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

    def calculate_streamline(self, data_container, streamline):
        next_dir = self._get_next_direction(streamline)
        next_dir, rot_matrix = self._apply_rot_matrix(next_dir)
        dwi, _ = self._get_dwi(data_container, streamline, rot_matrix=rot_matrix)
        if self.options.postprocessing is not None:
            dwi = self.options.postprocessing(dwi, data_container.data.b0,
                                              data_container.data.bvecs,
                                              data_container.data.bvals)
        if self.options.normalize:
            next_dir = (next_dir - self.options.normalize_mean)/self.options.normalize_std
        return (dwi, next_dir)

    def _get_dwi(self, data_container, streamline, rot_matrix=None):
        points = self._get_grid_points(streamline, rot_matrix=rot_matrix)
        return data_container.get_interpolated_dwi(points), points

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
        rot_matrix[0] = np.eye(3)
        for i in range(len(next_dir) - 1):
            rotation_from_vectors(rot_matrix[i + 1], reference, next_dir[i])
            next_dir[i] = rot_matrix[i].T @ next_dir[i]
        return next_dir, rot_matrix
        

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


class ClassificationProcessing(RegressionProcessing):
    def __init__(self, rotate=None, grid_dimension=None, grid_spacing=None, postprocessing=None,
                 sphere=None):

        RegressionProcessing.__init__(self, rotate=rotate, grid_dimension=grid_dimension,
                                      grid_spacing=grid_spacing, postprocessing=grid_spacing,
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
                   .format(rotate, sphere, *grid_dimension, grid_spacing, postprocessing.id))

    def calculate_streamline(self, data_container, streamline):
        dwi, next_dir = RegressionProcessing.calculate_streamline(self, data_container, streamline)
        sphere = self.sphere
        # code adapted from Benou "DeepTract",
        # https://github.com/itaybenou/DeepTract/blob/master/utils/train_utils.py
        sl_len = len(next_dir)
        l = len(sphere.theta) + 1
        classification_output = np.zeros((sl_len, l))
        for i in range(sl_len - 1):
            labels_odf = np.exp(-1 * sphere_distance(next_dir[i, :], np.asarray(
                [sphere.x, sphere.y, sphere.z]).T, radius=1, check_radius=False) * 10)
            classification_output[i][:-1] = labels_odf / np.sum(labels_odf)
            classification_output[i, -1] = 0.0

        classification_output[-1, -1] = 1 # stop condition
        return dwi, classification_output

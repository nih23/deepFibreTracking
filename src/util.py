"""Helpful functions required multiple times in different contexts

Methods
-------
rotation_from_vectors(rot, vector_orig, vector_fin)
    Calculates the rotation matrix required to rotate from one vector to another.
get_reference_orientation()
    Returns the reference orientation to use with this library. This could change, depending on config!
get_2D_sphere(no_phis=None, no_thetas=None)
    Retrieve evenly distributed 2D sphere out of phi and theta count
get_grid(grid_dimension)
    Calculates a (unrotated) grid for given dimensions
random_split(dataset, training_part=0.9)
    Splits a given dataset into train and validation part
"""
import torch
import numpy as np
from dipy.core.sphere import Sphere
from dipy.core.geometry import sphere_distance
from src.config import Config

def rotation_from_vectors(rot, vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    @param rot:           The 3x3 rotation matrix to update.
    @type rot:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)

    # The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = np.arccos(np.dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = np.cos(angle)
    sa = np.sin(angle)

    # Calculate the rotation matrix elements.
    rot[0, 0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    rot[0, 1] = -z*sa + (1.0 - ca)*x*y
    rot[0, 2] = y*sa + (1.0 - ca)*x*z
    rot[1, 0] = z*sa+(1.0 - ca)*x*y
    rot[1, 1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    rot[1, 2] = -x*sa+(1.0 - ca)*y*z
    rot[2, 0] = -y*sa+(1.0 - ca)*x*z
    rot[2, 1] = x*sa+(1.0 - ca)*y*z
    rot[2, 2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)

def get_reference_orientation():
    """Get current reference rotation
    
    Returns
    -------
    numpy.ndarray
        The reference rotation usable for rotations.
    """
    config = Config.get_config()
    orientation = config.get("DatasetOptions", "referenceOrientation", fallback="R+").upper()
    ref = None
    if orientation[0] == 'R':
        ref = np.array([1, 0, 0])
    elif orientation[0] == 'A':
        ref = np.array([0, 1, 0])
    elif orientation[0] == 'S':
        ref = np.array([0, 1, 0])
    if orientation[1] == '-':
        ref = ref * -1
    return ref

def get_2D_sphere(no_phis=None, no_thetas=None):
    """Retrieve evenly distributed 2D sphere out of phi and theta count.


    Parameters
    ----------
    no_phis : int, optional
        The numbers of phis in the sphere, by default as in config file / 16
    no_thetas : int, optional
        The numbers of thetas in the sphere, by default as in config file / 16

    Returns
    -------
    Sphere
        The 2D sphere requested
    """
    if no_thetas is None:
        no_thetas = Config.get_config().getint("2DSphereOptions", "noThetas", fallback="16")
    if no_phis is None:
        no_phis = Config.get_config().getint("2DSphereOptions", "noPhis", fallback="16")
    xi = np.arange(0, np.pi, (np.pi) / no_thetas) # theta
    yi = np.arange(-np.pi, np.pi, 2 * (np.pi) / no_phis) # phi

    basis = np.array(np.meshgrid(yi, xi))

    sphere = Sphere(theta=basis[0, :], phi=basis[1, :])

    return sphere

def get_grid(grid_dimension):
    """Calculates grid for given dimension

    Parameters
    ----------
    grid_dimension : numpy.ndarray
        The grid dimensions of the grid to calculate

    Returns
    -------
    numpy.ndarray
        The requested grid
    """
    (dx, dy, dz) = (grid_dimension - 1)/2
    return np.moveaxis(np.mgrid[-dx:dx+1, -dy:dy+1, -dz:dz+1], 0, 3)

def random_split(dataset, training_part=0.9):
    """Retrieves a dataset from given path and splits them randomly in train and test data.

    Parameters
    ----------
    dataset : Dataset
        The dataset to use
    training_part : float, optional
        The training part, by default 0.9 (90%)

    Returns
    -------
    tuple
        A tuple containing (train_dataset, validation_dataset)
    """
    train_len = int(training_part*len(dataset))
    test_len = len(dataset) - train_len
    (train_split, test_split) = torch.utils.data.random_split(dataset, (train_len, test_len))
    return train_split, test_split


def get_mask_from_lengths(lengths):
    """Returns a mask for given array of lengths
    
    Parameters
    ----------
    lenghts: Tensor
        The lengths to padd
    Returns
    -------
    Tensor
        The requested mask."""
    return (torch.arange(torch.max(lengths, device=lengths.device))[None, :] < lengths[:, None])

def apply_rotation_matrix_to_grid(grid, rot_matrix):
    """Applies the given list of rotation matrices to given grid

    Parameters
    ----------
    grid : numpy.ndarray
        The grid 
    rot_matrix : numpy.ndarray
        The rotation matrix with the dimensions (N, 3, 3)

    Returns
    -------
    numpy.ndarray
        The grid, rotated along the rotation_matrix; Shape: (N, ...grid_dimensions)
    """
    return (rot_matrix.repeat(grid.size/3, axis=0) @ grid[None, ].repeat(len(rot_matrix), axis=0).reshape(-1, 3, 1)).reshape((-1, *grid.shape))

def direction_to_classification(sphere, next_dir, include_stop=False, last_is_stop=False, stop_values=None):
    # code adapted from Benou "DeepTract",
    # https://github.com/itaybenou/DeepTract/blob/master/utils/train_utils.py

    sl_len = len(next_dir)
    loop_len = sl_len - 1 if include_stop and last_is_stop else sl_len
    l = len(sphere.theta) + 1 if include_stop else len(sphere.theta)
    classification_output = np.zeros((sl_len, l))
    for i in range(loop_len):
        if not (next_dir[i,0] == 0.0 and next_dir[i, 1] == 0.0 and next_dir[i, 2] == 0.0):
            labels_odf = np.exp(-1 * sphere_distance(next_dir[i, :], np.asarray(
                [sphere.x, sphere.y, sphere.z]).T, radius=1, check_radius=False) * 10)
            if include_stop:
                classification_output[i][:-1] = labels_odf / np.sum(labels_odf)
                classification_output[i, -1] = 0.0
            else:
                classification_output[i] = labels_odf / np.sum(labels_odf)
    if include_stop and last_is_stop:
        classification_output[-1, -1] = 1 # stop condition or
    if include_stop and stop_values is not None:
        classification_output[:,-1] = stop_values # stop values
    return classification_output
"""Helpful functions required multiple times in different contexts
"""
import torch
import numpy as np
from dipy.core.sphere import Sphere
from dipy.core.geometry import sphere_distance
from dipy.data import get_sphere

from .config import Config

def rotation_from_multiple_vectors(rot, vectors_orig, vectors_fin):
    """Calculates the rotation matrices required to rotate from one list of vectors to another.

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
    @param rot:           The Nx3x3 rotation matrix to update.
    @type rot:            Nx3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, dim Nx3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, dim Nx3
    """
    vectors_orig = vectors_orig / np.linalg.norm(vectors_orig, axis=1)[:, None]
    vectors_fin = vectors_fin / np.linalg.norm(vectors_fin, axis=1)[:, None]
    axes = np.cross(vectors_orig, vectors_fin)
    axes_lens = np.linalg.norm(axes, axis=1)

    axes_lens[axes_lens == 0] = 1

    axes = axes/axes_lens[:,None]

    x = axes[:,0]
    y = axes[:,1]
    z = axes[:,2]


    angles = np.arccos(np.sum(vectors_orig * vectors_fin, axis=1))
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles) # cos
    rot[:,0, 0] = 1.0 + (1.0 - cos_angles)*(x**2 - 1.0)
    rot[:,0, 1] = -z*sin_angles + (1.0 - cos_angles)*x*y
    rot[:,0, 2] = y*sin_angles + (1.0 - cos_angles)*x*z
    rot[:,1, 0] = z*sin_angles+(1.0 - cos_angles)*x*y
    rot[:,1, 1] = 1.0 + (1.0 - cos_angles)*(y**2 - 1.0)
    rot[:,1, 2] = -x*sin_angles+(1.0 - cos_angles)*y*z
    rot[:,2, 0] = -y*sin_angles+(1.0 - cos_angles)*x*z
    rot[:,2, 1] = x*sin_angles+(1.0 - cos_angles)*y*z
    rot[:,2, 2] = 1.0 + (1.0 - cos_angles)*(z**2 - 1.0)


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
    cos = np.cos(angle)
    sin = np.sin(angle)

    # Calculate the rotation matrix elements.
    rot[0, 0] = 1.0 + (1.0 - cos)*(x**2 - 1.0)
    rot[0, 1] = -z*sin + (1.0 - cos)*x*y
    rot[0, 2] = y*sin + (1.0 - cos)*x*z
    rot[1, 0] = z*sin+(1.0 - cos)*x*y
    rot[1, 1] = 1.0 + (1.0 - cos)*(y**2 - 1.0)
    rot[1, 2] = -x*sin+(1.0 - cos)*y*z
    rot[2, 0] = -y*sin+(1.0 - cos)*x*z
    rot[2, 1] = x*sin+(1.0 - cos)*y*z
    rot[2, 2] = 1.0 + (1.0 - cos)*(z**2 - 1.0)

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

def get_2d_sphere(no_phis=None, no_thetas=None):
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
    x_values = np.arange(0, np.pi, (np.pi) / no_thetas) # theta
    y_values = np.arange(-np.pi, np.pi, 2 * (np.pi) / no_phis) # phi

    basis = np.array(np.meshgrid(y_values, x_values))

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
    (delta_x, delta_y, delta_z) = (grid_dimension - 1)/2
    return np.moveaxis(np.mgrid[-delta_x:delta_x+1, -delta_y:delta_y+1, -delta_z:delta_z+1], 0, 3)

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
    lengths: Tensor
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
    return ((rot_matrix.repeat(grid.size/3, axis=0) @
             (grid[None, ].repeat(len(rot_matrix), axis=0).reshape(-1, 3, 1)))
            .reshape((-1, *grid.shape)))

def direction_to_classification(sphere, next_dir, include_stop=False,
    last_is_stop=False, stop_values=None):
    """
    Converts the directions into appropriate classification values for the given sphere.
    """
    # code adapted from Benou "DeepTract",exi
    # https://github.com/itaybenou/DeepTract/blob/master/utils/train_utils.py

    sl_len = len(next_dir)
    loop_len = sl_len - 1 if include_stop and last_is_stop else sl_len
    classification_len = len(sphere.theta) + 1 if include_stop else len(sphere.theta)
    classification_output = np.zeros((sl_len, classification_len))
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


def get_sphere_from_param(sphere, directions=None):
    "Given a sphere as either name or Sphere, returns a tuple of name, actual_sphere"
    sphere_name = "custom"
    if directions is not None:
        return sphere_name, Sphere(xyz=directions)
    if isinstance(sphere, Sphere):
        return sphere_name, sphere
    else:
        return sphere, get_sphere(sphere)

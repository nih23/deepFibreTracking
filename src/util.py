"""Helpful functions required multiple times"""
import numpy as np
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
    """Get current reference rotation"""
    config = Config.get_config()
    orientation = config.get("DatasetOptions", "referenceOrientation", fallback="R+").upper()
    ref = None
    if orientation[0] is 'R':
        ref = np.array([1, 0, 0])
    elif orientation[0] is 'A':
        ref = np.array([0, 1, 0])
    elif orientation[0] is 'S':
        ref = np.array([0, 1, 0])
    if orientation[1] is '-':
        ref = ref * -1
    return ref

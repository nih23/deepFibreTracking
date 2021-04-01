"""
The postprocessing submodule of the data module hosts different options
of postprocessing the DWI data. Those can be passed to datasets for further use.

Each postprocessing function returns a function to pass to the Dataset.
For unique identification, each of those functions contains an attribute `id`.
"""
import warnings

import numpy as np
from dipy.core.sphere import Sphere
from dipy.reconst.shm import real_sym_sh_mrtrix, smooth_pinv

from dfibert.config import Config
from dfibert.util import get_2d_sphere, get_sphere_from_param

def raw():
    """Does no resampling.

    Returns
    -------
    function
        A function with `id` attribute, which parses dwi according given params.
    """
    def _wrapper(dwi, _b0, _bvecs, _bvals):
        return dwi
    _wrapper.id = "raw"
    return _wrapper


def spherical_harmonics(sh_order=None, smooth=None):
    """Resamples the data using spherical harmonics

    The data is calculated out of the real DWI Sphere.

    Returns
    -------
    function
        A function with `id` attribute, which parses dwi accordingly.
    """
    config = Config.get_config()
    if sh_order is None:
        sh_order = config.getint("ResamplingOptions", "sphericalHarmonicsOrder", fallback="8")
    if smooth is None:
        smooth = config.getfloat("ResamplingOptions", "smooth", fallback="0.006")

    def _wrapper(dwi, _b0, bvecs, _bvals):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            # TODO - look if bvecs should be normalized, then this can be removed

            raw_sphere = Sphere(xyz=bvecs)

        real_sh, _, harmonics_order = real_sym_sh_mrtrix(sh_order, raw_sphere.theta, raw_sphere.phi)
        l = -harmonics_order * (harmonics_order + 1)
        inv_b = smooth_pinv(real_sh, np.sqrt(smooth) * l)
        data_sh = np.dot(dwi, inv_b.T)

        return data_sh
    _wrapper.id = "sh-order-{sh}-smooth-{sm}".format(sh=sh_order, sm=smooth)
    return _wrapper


def resample(directions=None, sh_order=None, smooth=None, mean_centering=None, sphere=None):
    """Resample the values according to given sphere or directions.

    The real sphere data is resampled to the new sphere, then spherical harmonics are applied.

    Returns
    -------
    function
        A function with `id` attribute, which parses dwi accordingly.
    """
    config = Config.get_config()
    if sh_order is None:
        sh_order = config.getint("ResamplingOptions", "sphericalHarmonicsOrder", fallback="8")
    if smooth is None:
        smooth = config.getfloat("ResamplingOptions", "smooth", fallback="0.006")
    if mean_centering is None:
        mean_centering = config.getboolean("ResamplingOptions", "meanCentering", fallback="no")
    if sphere is None:
        sphere = config.get("ResamplingOptions", "sphere", fallback="repulsion100")

    sphere_name, real_sphere = get_sphere_from_param(sphere, directions)
    real_sh, _, _ = real_sym_sh_mrtrix(sh_order, real_sphere.theta, real_sphere.phi)

    def _wrapper(dwi, b0_vals, bvecs, bvals):
        data_sh = spherical_harmonics(sh_order=sh_order, smooth=smooth)(dwi, b0_vals, bvecs, bvals)

        data_resampled = np.dot(data_sh, real_sh.T)

        if mean_centering:
            assert False # TODO should not be used
            idx = data_resampled.sum(axis=-1).nonzero()
            means = data_resampled[idx].mean(axis=0)
            data_resampled[idx] -= means
        return data_resampled
    _wrapper.id = ("resample-{sphere}-sh-order-{sh}-smooth-{sm}-mean_centering-{mc}"
                   .format(sphere=sphere_name, sh=sh_order, sm=smooth, mc=mean_centering))
    return _wrapper

def res100(sh_order=None, smooth=None, mean_centering=None):
    """Resamples the value to 100 directions with the repulsion100 sphere.

    Just a shortcut for the `resample` option.

    See Also
    --------
    resample: the function this is based on.

    Returns
    -------
    function
        A function with `id` attribute, which parses dwi accordingly.
    """
    return resample(sh_order=sh_order, smooth=smooth, mean_centering=mean_centering,
                    sphere="repulsion100")

def resample2d(sh_order=None, smooth=None, mean_centering=None, no_thetas=None, no_phis=None):
    """Resamples the value to directions with the 2D sphere.

    Just a shortcut for the `resample` option with 2D sphere.

    See Also
    --------
    resample: the function this is based on.
    dfibert.util.get_2d_sphere: the function the 2D sphere is generated with.

    Returns
    -------
    function
        A function with `id` attribute, which parses dwi accordingly.
    """
    func = resample(sh_order=sh_order, smooth=smooth, mean_centering=mean_centering,
                    sphere=get_2d_sphere(no_phis=no_phis, no_thetas=no_thetas))
    func.id = ("resample-2Dsphere-{nt}x{np}-sh-order-{sh}-smooth-{sm}-mean_centering-{mc}"
               .format(nt=no_thetas, np=no_phis, sh=sh_order, sm=smooth, mc=mean_centering))
    return func

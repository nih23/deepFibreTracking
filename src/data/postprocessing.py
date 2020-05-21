"""All postprocessing options for datasets"""

import numpy as np
from dipy.core.sphere import Sphere
from dipy.reconst.shm import real_sym_sh_mrtrix, smooth_pinv
from dipy.data import get_sphere

from src.config import Config
from src.util import get_2D_sphere

def raw():
    def _wrapper(dwi, _b0, _bvecs, _bvals):
        """raw data representation. Unnecessary for practical use, just here for completeness."""
        return dwi
    _wrapper.id = "raw"
    return _wrapper


def spherical_harmonics(sh_order=None, smooth=None):
    """Spherical Harmonics data representation"""
    config = Config.get_config()
    if sh_order is None:
        sh_order = config.getint("ResamplingOptions", "sphericalHarmonicsOrder", fallback="8")
    if smooth is None:
        smooth = config.getfloat("ResamplingOptions", "smooth", fallback="0.006")

    def _wrapper(dwi, _b0, bvecs, _bvals):
        raw_sphere = Sphere(xyz=bvecs)

        real_sh, _, n = real_sym_sh_mrtrix(sh_order, raw_sphere.theta, raw_sphere.phi)
        l = -n * (n + 1)
        inv_b = smooth_pinv(real_sh, np.sqrt(smooth) * l)
        data_sh = np.dot(dwi, inv_b.T)

        return data_sh
    _wrapper.id = "sh-order-{sh}-smooth-{sm}".format(sh=sh_order, sm=smooth)
    return _wrapper


def resample(directions=None, sh_order=None, smooth=None, mean_centering=None, sphere=None):
    """Resample the values according to given sphere"""
    config = Config.get_config()
    if sh_order is None:
        sh_order = config.getint("ResamplingOptions", "sphericalHarmonicsOrder", fallback="8")
    if smooth is None:
        smooth = config.getfloat("ResamplingOptions", "smooth", fallback="0.006")
    if mean_centering is None:
        mean_centering = config.getboolean("ResamplingOptions", "mean_centering", fallback="yes")
    if sphere is None:
        sphere = config.get("ResamplingOptions", "sphere", fallback="repulsion100")

    if isinstance(sphere, Sphere):
        rsphere = sphere
        sphere = "custom"
    else:
        rsphere = get_sphere(sphere)
    if directions is not None:
        rsphere = Sphere(xyz=directions)
    real_sh, _, _ = real_sym_sh_mrtrix(sh_order, rsphere.theta, rsphere.phi)

    def _wrapper(dwi, b0, bvecs, bvals):
        data_sh = spherical_harmonics(sh_order=sh_order, smooth=smooth)(dwi, b0, bvecs, bvals)

        data_resampled = np.dot(data_sh, real_sh.T)

        if mean_centering:
            idx = data_resampled.sum(axis=-1).nonzero()
            means = data_resampled[idx].mean(axis=0)
            data_resampled[idx] -= means
        return data_resampled
    _wrapper.id = ("resample-{sphere}-sh-order-{sh}-smooth-{sm}-mean_centering-{mc}"
                   .format(sphere=sphere, sh=sh_order, sm=smooth, mc=mean_centering))
    return _wrapper

def res100(sh_order=None, smooth=None, mean_centering=None):
    """Resample to 100 directions shortcut"""
    return resample(sh_order=sh_order, smooth=smooth, mean_centering=mean_centering,
                    sphere="repulsion100")

def resample2D(sh_order=None, smooth=None, mean_centering=None, no_thetas=None, no_phis=None):
    """Resample to 2D Sphere"""
    func = resample(sh_order=sh_order, smooth=smooth, mean_centering=mean_centering,
                    sphere=get_2D_sphere(no_phis=no_phis, no_thetas=no_thetas))
    func.id = ("resample-2Dsphere-{nt}x{np}-sh-order-{sh}-smooth-{sm}-mean_centering-{mc}"
               .format(nt=no_thetas, np=no_phis, sh=sh_order, sm=smooth, mc=mean_centering))
    return func
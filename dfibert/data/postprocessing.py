"""
The postprocessing submodule of the data module hosts different options
of postprocessing the DWI data. Those can be passed to datasets for further use.

Each postprocessing function returns a function to pass to the Dataset.
For unique identification, each of those functions contains an attribute `id`.
"""
from typing import Union
import numpy as np
from dipy.core.sphere import Sphere
from dipy.reconst.shm import real_sym_sh_mrtrix, smooth_pinv
from dipy.data import get_sphere

from dfibert.util import get_2D_sphere


class PostprocessingOption(object):
    def process(self, data_container, points, dwi):
        raise NotImplementedError()


class Raw(PostprocessingOption):
    """Does no resampling."""
    def process(self, data_container, points, dwi):
        return dwi


class SphericalHarmonics(PostprocessingOption):
    def __init__(self, sh_order=8, smooth=0.006):
        """
        Resamples the data using spherical harmonics

        The resampled data is calculated using the DWI Sphere.

        :param sh_order: The order of the spherical harmonics
        :param smooth:
        """
        super().__init__()
        self.sh_order = sh_order
        self.smooth = smooth

    def process(self, data_container, points, dwi):
        raw_sphere = Sphere(xyz=data_container.bvecs)

        real_sh, _, n = real_sym_sh_mrtrix(self.sh_order, raw_sphere.theta, raw_sphere.phi)
        l = -n * (n + 1)
        inv_b = smooth_pinv(real_sh, np.sqrt(self.smooth) * l)
        data_sh = np.dot(dwi, inv_b.T)

        return data_sh


class Resample(SphericalHarmonics):
    def __init__(self, sh_order=8, smooth=0.006, sphere: Union[Sphere, str] = "repulsion100"):
        """
        Resample the values according to given sphere or directions.

        The real sphere data is resampled to the new sphere, then spherical harmonics are applied.

        :param sh_order: The order of the spherical harmonics
        :param smooth:
        :param sphere: The sphere we are resampling to
        """
        super().__init__(sh_order=sh_order, smooth=smooth)
        if isinstance(sphere, Sphere):
            self.sphere = Sphere
        else:  # get with name
            self.sphere = get_sphere(sphere)
        self.real_sh, _, _ = real_sym_sh_mrtrix(self.sh_order, self.sphere.theta, self.sphere.phi)

    def process(self, data_container, points, dwi):
        data_sh = super().process(data_container, points, dwi)
        data_resampled = np.dot(data_sh, self.real_sh.T)

        return data_resampled


class Resample100(Resample):
    def __init__(self, sh_order=8, smooth=0.006):
        """
        Resamples the value to 100 directions with the repulsion100 sphere.

        Just a shortcut for the `resample` option.
        :param sh_order: The order of the spherical harmonics
        :param smooth:
        """
        super().__init__(sh_order=sh_order, smooth=smooth, sphere="repulsion100")


class Resample2D(Resample):
    def __init__(self, sh_order=8, smooth=0.006, no_thetas=16, no_phis=16):
        """
        Resamples the value to directions with the 2D sphere.
        Just a shortcut for the `resample` option with 2D sphere.

        See `dfibert.util.get_2D_sphere` for more details on how the 2D sphere is generated.

        :param sh_order: The order of the spherical harmonics
        :param smooth:
        :param no_thetas: the number of thetas to use for the sphere generation
        :param no_phis: the number of thetas to use for the sphere generation
        """
        super().__init__(sh_order=sh_order, smooth=smooth, sphere=get_2D_sphere(no_phis=no_phis, no_thetas=no_thetas))
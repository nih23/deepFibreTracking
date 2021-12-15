"""Implementing different tracking approaches"""

import os
from typing import List

from dipy.core.gradients import gradient_table
from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking import metrics
from dipy.reconst.dti import TensorModel
from dipy.io.streamline import save_vtk_streamlines, load_vtk_streamlines
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.data import get_sphere, default_sphere
from dipy.direction import peaks_from_model, DeterministicMaximumDirectionGetter
import dipy.reconst.dti as dti


def _get_seeds(data_container, random_seeds=False, seeds_count=30000, seeds_per_voxel=False):
    if not random_seeds:
        return seeds_from_mask(data_container.binary_mask, affine=data_container.aff)
    else:
        return random_seeds_from_mask(data_container.binary_mask,
                                      seeds_count=seeds_count,
                                      seed_count_per_voxel=seeds_per_voxel,
                                      affine=data_container.aff)


def get_csd_streamlines(data_container, random_seeds=False, seeds_count=30000, seeds_per_voxel=False, step_width=1.0,
                        roi_r=10, auto_response_fa_threshold=0.7, fa_threshold=0.15, relative_peak_threshold=0.5,
                        min_separation_angle=25):
    """
    Tracks and returns CSD Streamlines for the given DataContainer.

    Parameters
    ----------
    data_container
        The DataContainer we would like to track streamlines on
    random_seeds
        A boolean indicating whether we would like to use random seeds
    seeds_count
        If we use random seeds, this specifies the seed count
    seeds_per_voxel
        If True, the seed count is specified per voxel
    step_width
        The step width used while tracking
    roi_r
        The radii of the cuboid roi for the automatic estimation of single-shell single-tissue response function using FA.
    auto_response_fa_threshold
        The FA threshold for the automatic estimation of single-shell single-tissue response function using FA.
    fa_threshold
        The FA threshold to use to stop tracking
    relative_peak_threshold
        The relative peak threshold to use to get peaks from the CSDModel
    min_separation_angle
        The minimal separation angle of peaks
    Returns
    -------
    Streamlines
        A list of Streamlines
    """
    seeds = _get_seeds(data_container, random_seeds, seeds_count, seeds_per_voxel)

    gtab = gradient_table(data_container.bvals, data_container.bvecs)
    response, _ = auto_response_ssst(gtab, data_container.dwi, roi_radii=roi_r, fa_thr=auto_response_fa_threshold)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response)

    direction_getter = peaks_from_model(model=csd_model,
                                        data=data_container.dwi,
                                        sphere=get_sphere('symmetric724'),
                                        mask=data_container.binary_mask,
                                        relative_peak_threshold=relative_peak_threshold,
                                        min_separation_angle=min_separation_angle,
                                        parallel=False)

    dti_fit = dti.TensorModel(gtab, fit_method='LS').fit(data_container.dwi, mask=data_container.binary_mask)
    classifier = ThresholdStoppingCriterion(dti_fit.fa, fa_threshold)

    streamlines_generator = LocalTracking(direction_getter, classifier, seeds, data_container.aff, step_size=step_width)
    streamlines = Streamlines(streamlines_generator)

    return streamlines


def get_dti_streamlines(data_container, random_seeds=False, seeds_count=30000, seeds_per_voxel=False, step_width=1.0,
                        max_angle=30.0, fa_threshold=0.15):
    """
    Tracks and returns CSD Streamlines for the given DataContainer.

    Parameters
    ----------
    data_container
        The DataContainer we would like to track streamlines on
    random_seeds
        A boolean indicating whether we would like to use random seeds
    seeds_count
        If we use random seeds, this specifies the seed count
    seeds_per_voxel
        If True, the seed count is specified per voxel
    step_width
        The step width used while tracking
    fa_threshold
        The FA threshold to use to stop tracking
    max_angle
        The maximum allowed angle between incoming and outgoing angle, float between 0.0 and 90.0 deg
    Returns
    -------
    Streamlines
        A list of Streamlines
    """
    seeds = _get_seeds(data_container, random_seeds, seeds_count, seeds_per_voxel)

    gtab = gradient_table(data_container.bvals, data_container.bvecs)

    dti_fit = TensorModel(gtab).fit(data_container.dwi, mask=data_container.binary_mask)
    dti_fit_odf = dti_fit.odf(sphere=default_sphere)

    direction_getter = DeterministicMaximumDirectionGetter.from_pmf(dti_fit_odf,
                                                                    max_angle=max_angle,
                                                                    sphere=default_sphere)
    classifier = ThresholdStoppingCriterion(dti_fit.fa, fa_threshold)

    streamlines_generator = LocalTracking(direction_getter, classifier, seeds, data_container.aff, step_size=step_width)
    streamlines = Streamlines(streamlines_generator)

    return streamlines


def save_streamlines(streamlines: list, path: str, to_lps=True, binary=False):
    """
    Saves the given streamlines to a file
    Parameters
    ----------
    streamlines
        The streamlines we want to save
    path
        The path we save the streamlines to
    to_lps
        A boolean indicating whether we want to save them in the LPS format instead of RAS (True by default)
    binary
        If True, the file will be written in a binary format.
    Returns
    -------

    """
    save_vtk_streamlines(streamlines, path, to_lps=to_lps, binary=binary)


def load_streamlines(path: str, to_lps=True) -> list:
    """
    Loads streamlines from the given path.
    Parameters
    ----------
    path
        The path to load streamlines from
    to_lps
        If True, we load streamlines under the assumption that they were stored in LPS (True by default)
    Returns
    -------
    list
        The streamlines we are trying to load
    """
    if os.path.isdir(path):
        streamlines = []
        for file in os.listdir(path):
            if file.endswith(".fib") or file.endswith(".vtk"):
                sl = load_vtk_streamlines(os.path.join(path, file), to_lps)
                streamlines.extend(sl)
    else:
        streamlines = load_vtk_streamlines(path, to_lps)
    return streamlines


def filtered_streamlines_by_length(streamlines: List, minimum=20, maximum=200) -> List:
    """
    Returns filtered streamlines that are longer than minimum (in mm) and shorter than maximum (in mm)
    Parameters
    ----------
    streamlines
        The streamlines we would like to filter
    minimum
        The minimum length in mm
    maximum
        The maximum length in mm
    Returns
    -------
    List
        The filtered streamlines
    """
    return [x for x in streamlines if minimum <= metrics.length(x) <= maximum]

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


"""class CSDTracker(object):
    def __init__(self, step_width=1.0, roi_r=10, auto_response_fa_threshold=0.7, fa_threshold=0.7,
                 relative_peak_threshold=0.5, min_separation_angle=25):
        self.step_width = step_width
        self.roi_r = roi_r
        self.auto_response_fa_threshold = auto_response_fa_threshold
        self.fa_threshold = fa_threshold
        self.relative_peak_threshold = relative_peak_threshold
        self.min_separation_angle = min_separation_angle

    def track(self, data_container, random_seeds=False, seeds_count=30000, seeds_per_voxel=False) -> Streamlines:
        seeds = _get_seeds(data_container, random_seeds, seeds_count, seeds_per_voxel)

        gtab = gradient_table(data_container.bvals, data_container.bvecs)
        response, _ = auto_response_ssst(gtab, data_container.dwi, roi_radii=self.roi_r,
                                         fa_thr=self.auto_response_fa_threshold)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response)

        direction_getter = peaks_from_model(model=csd_model,
                                            data=data_container.dwi,
                                            sphere=get_sphere('symmetric724'),
                                            mask=data_container.binary_mask,
                                            relative_peak_threshold=self.relative_peak_threshold,
                                            min_separation_angle=self.min_separation_angle,
                                            parallel=False)
    
        dti_fit = dti.TensorModel(gtab, fit_method='LS').fit(data_container.dwi, mask=data_container.binary_mask)

        classifier = ThresholdStoppingCriterion(dti_fit.fa, self.fa_threshold)
        streamlines_generator = LocalTracking(direction_getter, classifier, seeds, data_container.aff,
                                              step_size=self.step_width)
        streamlines = Streamlines(streamlines_generator)

        return streamlines
"""


def get_csd_streamlines(data_container, random_seeds=False, seeds_count=30000, seeds_per_voxel=False, step_width=1.0,
                        roi_r=10, auto_response_fa_threshold=0.7, fa_threshold=0.15, relative_peak_threshold=0.5,
                        min_separation_angle=25):
    """

    Parameters
    ----------
    data_container
    random_seeds
    seeds_count
    seeds_per_voxel
    step_width
    roi_r
    auto_response_fa_threshold
    fa_threshold
    relative_peak_threshold
    min_separation_angle

    Returns
    -------

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


def save_streamlines(streamlines: list, filename: str, to_lps=True, binary=False):
    save_vtk_streamlines(streamlines, filename, to_lps=to_lps, binary=binary)


def load_streamlines(path: str, to_lps=True):
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
    returns filtered streamlines that are longer than minimum (in mm) and shorter than maximum (in mm)
    """
    return [x for x in streamlines if minimum <= metrics.length(x) <= maximum]



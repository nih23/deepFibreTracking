"""Implementing different tracker classes"""

import os
import random

from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking import metrics
from dipy.reconst.dti import TensorModel
from dipy.io.streamline import save_vtk_streamlines, load_vtk_streamlines
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response
from dipy.data import get_sphere, default_sphere
from dipy.direction import peaks_from_model, DeterministicMaximumDirectionGetter
import dipy.reconst.dti as dti

from src.data import Object
from src.config import Config
from src.cache import Cache

class Error(Exception):
    """Base class for Tracker exceptions."""

    def __init__(self, msg=''):
        self.message = msg
        Exception.__init__(self, msg)

    def __repr__(self):
        return self.message

    __str__ = __repr__

class StreamlinesAlreadyTrackedError(Error):
    """Error thrown if streamlines are already tracked."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.data_container = tracker.data_container
        Error.__init__(self, msg=("There are already {sl} streamlines tracked out of dataset {id}. "
                                  "Create a new Tracker object to change parameters.")
                       .format(sl=len(tracker.streamlines), id=self.data_container.id))

class ISMRMStreamlinesNotCorrectError(Error):
    """Error thrown if streamlines are already tracked."""

    def __init__(self, tracker, path):
        self.tracker = tracker
        self.path = path
        Error.__init__(self, msg=("The streamlines located in {path} do not match the "
                                  "ISMRM 2015 Ground Truth Streamlines.").format(path=path))

class StreamlinesNotTrackedError(Error):
    """Error thrown if streamlines weren't tracked yet."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.data_container = tracker.data_container
        Error.__init__(self, msg=("The streamlines weren't tracked yet from Dataset {id}. "
                                  "Call Tracker.track() to track the streamlines.")
                       .format(id=self.data_container.id))
class Tracker():
    """Universal Tracker class"""
    def __init__(self, data_container):
        self.id = str(self.__class__.__name__)
        if data_container is not None:
            self.data_container = data_container
            self.id = self.id + "-" + str(data_container.id)
        self.streamlines = None
    def track(self):
        """Track given data"""
        if self.streamlines is not None:
            raise StreamlinesAlreadyTrackedError(self) from None
        if Cache.get_cache().in_cache(self.id):
            self.streamlines = Cache.get_cache().get(self.id)

    def get_streamlines(self):
        """Retrieve the calculated streamlines"""
        if self.streamlines is None:
            raise StreamlinesNotTrackedError(self) from None
        return self.streamlines

class SeedBasedTracker(Tracker):
    """Seed based tracker"""
    def __init__(self, data_container,
                 random_seeds=None,
                 seeds_count=None,
                 seeds_per_voxel=None,
                 step_width=None,
                 min_length=None,
                 max_length=None):
        Tracker.__init__(self, data_container)
        self.options = Object()
        if seeds_count is not None and random_seeds is None:
            random_seeds = True
        if random_seeds is None:
            random_seeds = Config.get_config().getboolean("tracking", "randomSeeds", fallback="no")
        if seeds_count is None:
            seeds_count = Config.get_config().getint("tracking", "seedsCount", fallback="30000")
        if seeds_per_voxel is None:
            seeds_per_voxel = Config.get_config().getboolean("tracking", "seedsPerVoxel",
                                                             fallback="no")
        if step_width is None:
            step_width = Config.get_config().getfloat("tracking", "stepWidth", fallback="1.0")
        if min_length is None:
            min_length = Config.get_config().getfloat("tracking", "minimumStreamlineLength",
                                                      fallback="20")
        if max_length is None:
            max_length = Config.get_config().getfloat("tracking", "maximumStreamlineLength",
                                                      fallback="200")
        if random_seeds:
            self.id = self.id + \
                      "-randomStreamlines-no{n}-perVoxel{perVoxel}".format(n=seeds_count,
                                                                           perVoxel=seeds_per_voxel)
        self.id = self.id + "-sw{ss}-mil{mil}-max{mal}".format(ss=step_width,
                                                               mil=min_length,
                                                               mal=max_length)
        self.data = data_container.data
        self.seeds = None
        self.options.seeds_count = seeds_count
        self.options.seeds_per_voxel = seeds_per_voxel
        self.options.random_seeds = random_seeds
        self.options.max_length = max_length
        self.options.min_length = min_length
        self.options.step_width = step_width
    def _track(self, classifier, direction_getter):
        if self.streamlines is not None:
            raise StreamlinesAlreadyTrackedError(self) from None
        streamlines_generator = LocalTracking(direction_getter, classifier, self.seeds,
                                              self.data.aff, step_size=self.options.step_width)
        streamlines = Streamlines(streamlines_generator)
        streamlines = self.filter_streamlines_by_length(streamlines,
                                                        minimum=self.options.min_length,
                                                        maximum=self.options.max_length)
        self.streamlines = streamlines
    def track(self):
        Tracker.track(self)
        if self.streamlines is None:
            if not self.options.random_seeds:
                seeds = seeds_from_mask(self.data.binarymask, affine=self.data.aff)
            else:
                seeds = random_seeds_from_mask(self.data.binarymask,
                                               seeds_count=self.options.seeds_count,
                                               seed_count_per_voxel=self.options.seeds_per_voxel,
                                               affine=self.data.aff)
            self.seeds = seeds

    def save_to_file(self, path):
        """Save the calculated streamlines to file"""
        if self.streamlines is None:
            raise StreamlinesNotTrackedError(self) from None
        save_vtk_streamlines(self.streamlines, path)

    def filter_streamlines_by_length(self, streamlines,
                                     minimum=Config.get_config()
                                     .getfloat("tracking", "minimumStreamlineLength",
                                               fallback="20"),
                                     maximum=Config.get_config()
                                     .getfloat("tracking", "maximumStreamlineLength",
                                               fallback="200")):
        """
        removes streamlines that are shorter than minimumLength (in mm)
        """
        return [x for x in streamlines if metrics.length(x) > minimum
                and metrics.length(x) < maximum]

class CSDTracker(SeedBasedTracker):
    """A CSD based Tracker"""
    def __init__(self, data_container,
                 random_seeds=None,
                 seeds_count=None,
                 seeds_per_voxel=None,
                 step_width=None,
                 min_length=None,
                 max_length=None,
                 fa_threshold=None):
        SeedBasedTracker.__init__(self, data_container, random_seeds, seeds_count, seeds_per_voxel,
                                  step_width, min_length, max_length)
        if fa_threshold is None:
            fa_threshold = Config.get_config().getfloat("tracking", "faTreshhold", fallback="0.15")
        self.id = self.id + "-fa{fa}".format(fa=fa_threshold)
        self.options.fa_threshold = fa_threshold

    def track(self):
        SeedBasedTracker.track(self)
        if self.streamlines is not None:
            return
        roi_r = Config.get_config().getint("CSDTracking", "autoResponseRoiRadius",
                                           fallback="10")
        fa_thr = Config.get_config().getfloat("CSDTracking", "autoResponseFaThreshold",
                                              fallback="0.7")
        response, _ = auto_response(self.data.gtab, self.data.dwi, roi_radius=roi_r, fa_thr=fa_thr)
        csd_model = ConstrainedSphericalDeconvModel(self.data.gtab, response)
        relative_peak_thr = Config.get_config().getfloat("CSDTracking", "relativePeakTreshold",
                                                         fallback="0.5")
        min_separation_angle = Config.get_config().getfloat("CSDTracking", "minimumSeparationAngle",
                                                            fallback="25")
        direction_getter = peaks_from_model(model=csd_model,
                                            data=self.data.dwi,
                                            sphere=get_sphere('symmetric724'),
                                            mask=self.data.binarymask,
                                            relative_peak_threshold=relative_peak_thr,
                                            min_separation_angle=min_separation_angle,
                                            parallel=False)
        dti_fit = dti.TensorModel(self.data.gtab, fit_method='LS')
        dti_fit = dti_fit.fit(self.data.dwi, mask=self.data.binarymask)
        self._track(ThresholdStoppingCriterion(dti_fit.fa, self.options.fa_threshold),
                    direction_getter)
        Cache.get_cache().set(self.id, self.streamlines)

class DTITracker(SeedBasedTracker):
    """A DTI based Tracker"""
    def __init__(self, data_container,
                 random_seeds=None,
                 seeds_count=None,
                 seeds_per_voxel=None,
                 step_width=None,
                 min_length=None,
                 max_length=None,
                 fa_threshold=None):
        SeedBasedTracker.__init__(self, data_container, random_seeds, seeds_count, seeds_per_voxel,
                                  step_width, min_length, max_length)
        if fa_threshold is None:
            fa_threshold = Config.get_config().getfloat("tracking", "faTreshhold", fallback="0.15")
        self.id = self.id + "-fa{fa}".format(fa=fa_threshold)
        self.options.fa_threshold = fa_threshold

    def track(self):
        SeedBasedTracker.track(self)
        if self.streamlines is not None:
            return
        dti_model = TensorModel(self.data.gtab)
        dti_fit = dti_model.fit(self.data.dwi, mask=self.data.binarymask)
        dti_fit_odf = dti_fit.odf(sphere=default_sphere)
        max_angle = Config.get_config().getfloat("DTITracking", "maxAngle", fallback="30.0")
        direction_getter = DeterministicMaximumDirectionGetter.from_pmf(dti_fit_odf,
                                                                        max_angle=max_angle,
                                                                        sphere=default_sphere)
        self._track(ThresholdStoppingCriterion(dti_fit.fa, self.options.fa_threshold),
                    direction_getter)
        Cache.get_cache().set(self.id, self.streamlines)

class StreamlinesFromFileTracker(Tracker):
    """A Tracker class representing preloaded Streamlines from file."""
    def __init__(self, path):
        Tracker.__init__(self, None)
        self.path = path
        self.id = self.id + "-" + path.replace(os.path.sep, "+")

    def track(self):
        Tracker.track(self)
        self.streamlines = load_vtk_streamlines(self.path)

class ISMRMReferenceStreamlinesTracker(Tracker):
    """Class representing the ISMRM 2015 Ground Truth fiber tracks."""
    def __init__(self, data_container, streamline_count=None):
        Tracker.__init__(self, data_container)
        self.options = Object()
        self.options.streamline_count = streamline_count
        if streamline_count is not None:
            self.id = self.id + "-" + str(streamline_count)
        self.path = Config.get_config().get("data", "pathISMRMGroundTruth",
                                            fallback='data/ISMRM2015GroundTruth')
        self.path = self.path.rstrip(os.path.sep)


    def track(self):
        Tracker.track(self)
        self.streamlines = []
        bundle_count = 0
        for file in os.listdir(self.path):
            if file.endswith(".fib"):
                bundle_count = bundle_count + 1
                sl = load_vtk_streamlines(os.path.join(self.path, file))
                self.streamlines.extend(sl)
        if len(self.streamlines) != 200433 or bundle_count != 25:
            raise ISMRMStreamlinesNotCorrectError(self, self.path)
        if self.options.streamline_count is not None:
            self.streamlines = random.sample(self.streamlines, self.options.streamline_count)

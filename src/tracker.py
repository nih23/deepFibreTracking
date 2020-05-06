"""Implementing different tracker classes"""
from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking import metrics
from dipy.io.streamline import save_vtk_streamlines
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
import dipy.reconst.dti as dti

from src.data import Object
from src.config import Config


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
                                  .format(sl=len(tracker.streamlines), id=self.data_container.id),
                                  "Create a new Tracker object to change parameters."))
class StreamlinesNotTrackedError(Error):
    """Error thrown if streamlines weren't tracked yet."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.data_container = tracker.data_container
        Error.__init__(self, msg=("The streamlines weren't tracked yet from Dataset {id}. "
                                  .format(id=self.data_container.id),
                                  "Call Tracker.track() to track the streamlines."))
class Tracker():
    """Universal Tracker class"""
    def __init__(self, data_container):
        self.data_container = data_container
    def track(self):
        """Track given data"""


class SeedBasedTracker(Tracker):
    """Seed based tracker"""
    def __init__(self, data_container,
                 random_seeds=None,
                 seeds_count=None,
                 seeds_per_voxel=None,
                 step_size=None,
                 min_length=None,
                 max_length=None):
        Tracker.__init__(self, data_container)
        self.options = Object()
        if random_seeds is None:
            random_seeds = Config.get_config().getboolean("tracking", "randomSeeds", fallback="no")
        if seeds_count is None:
            seeds_count = Config.get_config().getint("tracking", "seedsCount", fallback="30000")
        if seeds_per_voxel is None:
            seeds_per_voxel = Config.get_config().getboolean("tracking", "seedsPerVoxel",
                                                             fallback="no")
        if step_size is None:
            step_size = Config.get_config().getfloat("tracking", "stepSize", fallback="1.0")
        if min_length is None:
            min_length = Config.get_config().getfloat("tracking", "minimumStreamlineLength",
                                                      fallback="20")
        if max_length is None:
            max_length = Config.get_config().getfloat("tracking", "maximumStreamlineLength",
                                                      fallback="200")
        self.streamlines = None
        self.data = data_container.data
        if not random_seeds:
            seeds = seeds_from_mask(self.data.binarymask, affine=self.data.aff)
        else:
            seeds = random_seeds_from_mask(self.data.binarymask, seeds_count=seeds_count,
                                           seed_count_per_voxel=seeds_per_voxel,
                                           affine=self.data.aff)
        self.seeds = seeds
        self.options.max_length = max_length
        self.options.min_length = min_length
        self.options.step_size = step_size
    def _track(self, classifier, direction_getter):
        if self.streamlines is not None:
            raise StreamlinesAlreadyTrackedError(self) from None
        streamlines_generator = LocalTracking(direction_getter, classifier, self.seeds,
                                              self.data.aff, step_size=self.options.step_size)
        streamlines = Streamlines(streamlines_generator)
        streamlines = self.filter_streamlines_by_length(streamlines,
                                                        minimum=self.options.min_length,
                                                        maximum=self.options.max_length)
        self.streamlines = streamlines

    def get_streamlines(self):
        """Retrieve the calculated streamlines"""
        if self.streamlines is None:
            raise StreamlinesNotTrackedError(self) from None
        return self.streamlines

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
                 step_size=None,
                 min_length=None,
                 max_length=None,
                 fa_threshold=None):
        SeedBasedTracker.__init__(self, data_container, random_seeds, seeds_count, seeds_per_voxel,
                                  step_size, min_length, max_length)
        if fa_threshold is None:
            fa_threshold = Config.get_config().getfloat("tracking", "faTreshhold", fallback="0.15")
        self.options.fa_threshold = fa_threshold

    def track(self):
        response, _ = auto_response(self.data.gtab, self.data.dwi, roi_radius=10, fa_thr=0.7)
        csd_model = ConstrainedSphericalDeconvModel(self.data.gtab, response)
        direction_getter = peaks_from_model(model=csd_model,
                                            data=self.data.dwi,
                                            sphere=get_sphere('symmetric724'),
                                            mask=self.data.binarymask,
                                            relative_peak_threshold=.5,
                                            min_separation_angle=25,
                                            parallel=False)
        dti_fit = dti.TensorModel(self.data.gtab, fit_method='LS')
        dti_fit = dti_fit.fit(self.data.dwi, mask=self.data.binarymask)
        self._track(ThresholdStoppingCriterion(dti_fit.fa, self.options.fa_threshold),
                    direction_getter)

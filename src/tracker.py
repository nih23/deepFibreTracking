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

from src.config import Config


class Tracker():
    def __init__(self, data_container):
        pass


class SeedBasedTracker(Tracker):
    def __init__(self, data_container,
                 random_seeds=None,
                 seeds_count=None,
                 seeds_per_voxel=None,
                 step_size=None,
                 min_length=None,
                 max_length=None):
        if random_seeds is None:
            random_seeds = Config.get_config().getboolean("tracking", "useRandomSeeds", fallback="no")
        if seeds_count is None:
            seeds_count = Config.get_config().getint("tracking", "seedsCount", fallback="30000")
        if seeds_per_voxel is None:
            seeds_per_voxel = Config.get_config().getboolean("tracking", "seedsPerVoxel", fallback="no")
        if step_size is None:
            step_size = Config.get_config().getfloat("tracking", "stepSize", fallback="1.0")
        if minimum_sl_length is None:
            minimum_sl_length = Config.get_config().getfloat("tracking", "minimumStreamlineLength", fallback="20")
        if maximum_sl_length is None:
            maximum_sl_length = Config.get_config().getfloat("tracking", "maximumStreamlineLength", fallback="200")
        self.streamlines = None
        (_, _, gtab, _, aff, _, _, binarymask) = data_container.data
        if not random_seeds:
            seeds = seeds_from_mask(binarymask, affine=aff)
        else:
            seeds = random_seeds_from_mask(binarymask, seeds_count=seeds_count, 
                                           seed_count_per_voxel=seeds_per_voxel, affine=aff)
        self.seeds = seeds
        self.mask = binarymask
        self.gtab = gtab
        self.max_length = max_length
        self.min_length = min_length
        self.step_size = step_size
        self.aff = aff
        self.data = data_container.data
    def track(self, classifier, directionGetter):
        if self.streamlines is not None:
            pass # TODO Error
        streamlines_generator = LocalTracking(directionGetter, classifier, self.seeds, self.aff,
                                              step_size=self.step_size)
        streamlines = Streamlines(streamlines_generator)
        streamlines = self.filterStreamlinesByLength(streamlines, minimumLength=self.min_length, 
                                                     maximumLength=self.max_length)
        self.streamlines = streamlines

    def get_streamlines(self):
        if self.streamlines is None:
            pass # TODO Error
        return self.streamlines

    def saveToFile(self, path):
        if self.streamlines is None:
            pass # TODO Error
        save_vtk_streamlines(self.streamlines, path)

    def filterStreamlinesByLength(self, streamlines, 
                                  minimumLength=Config.get_config().getfloat("tracking", "minimumStreamlineLength",fallback="20"),
                                  maximumLength=Config.get_config().getfloat("tracking", "maximumStreamlineLength", fallback="200")):
        """
        removes streamlines that are shorter than minimumLength (in mm)
        """
        return [x for x in streamlines if metrics.length(x) > minimumLength and metrics.length(x) < maximumLength]

class CSDTracker(SeedBasedTracker):
    def __init__(self, data_container,
                 random_seeds=None,
                 seeds_count=None,
                 seeds_per_voxel=None,
                 step_size=None,
                 min_length=None,
                 max_length=None,
                 fa_threshold=None):
        SeedBasedTracker.__init__(self, data_container, random_seeds, seeds_count, seeds_per_voxel, step_size, min_length, max_length)
        if fa_threshold is None:
            fa_threshold = Config.get_config().getfloat("tracking", "faTreshhold", fallback="0.15")
        self.fa_threshold = fa_threshold

    def track(self):
        (_, _, gtab, dwi, _, _, _, binarymask) = self.data
        response, _ = auto_response(gtab, dwi, roi_radius=10, fa_thr=0.7)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response)
        directionGetter = peaks_from_model(model=csd_model,
                                     data=dwi,
                                     sphere=get_sphere('symmetric724'),
                                     mask=binarymask,
                                     relative_peak_threshold=.5,
                                     min_separation_angle=25,
                                     parallel=False)
        dti_fit = dti.TensorModel(gtab, fit_method='LS').fit(dwi, mask=binarymask)
        SeedBasedTracker.track(self, ThresholdStoppingCriterion(dti_fit.fa, self.fa_threshold), directionGetter)
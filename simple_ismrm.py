from src.tracker import SeedBasedTracker, StreamlinesAlreadyTrackedError
from src.config import Config
from src.cache import Cache
from src.util import get_reference_orientation, rotation_from_vectors, get_grid
import numpy as np
class ModelTracker(SeedBasedTracker):
    """A CSD based Tracker"""
    def __init__(self, data_container,
                 model,
                 random_seeds=None,
                 seeds_count=None,
                 seeds_per_voxel=None,
                 step_width=None,
                 min_length=None,
                 max_length=None,
                 fa_threshold=None):
        SeedBasedTracker.__init__(self, data_container, random_seeds, seeds_count, seeds_per_voxel,
                                  step_width, min_length, max_length)
        self.model = model
        if fa_threshold is None:
            fa_threshold = Config.get_config().getfloat("tracking", "faTreshhold", fallback="0.15")
        self.id = self.id + "-fa{fa}".format(fa=fa_threshold)
        self.options.fa_threshold = fa_threshold
    def _track(self):
        if self.streamlines is not None:
            raise StreamlinesAlreadyTrackedError(self) from None

        streamlines = self._trackWithModel()
        streamlines = self.filter_streamlines_by_length(streamlines,
                                                        minimum=self.options.min_length,
                                                        maximum=self.options.max_length)
    def _trackWithModel(self, no_iterations=200, batch_size=65536):
        model = self.model.cuda()
        data_container = self.data_container
        seeds = self.seeds
        grid = get_grid((7,3,3))
        streamlines = np.zeros((2* len(seeds), no_iterations + 1, 3))
        streamlines[:len(seeds), 0] = seeds
        streamlines[len(seeds):,0] = seeds
        for idx in range(0, len(seeds), batch_size):
            model.reset()
            idx_range = range(idx, min(idx+batch_size, len(seeds)))    
            idx_range_back = range(idx + len(seeds), min(idx+batch_size, len(seeds)) + len(seeds))

            #First prediction
            points = streamlines[idx_range, 0]
            points_w_grid = points[:, None, None, None, :] + grid
            dwi = data_container.get_interpolated_dwi(points_w_grid)
            print(dwi.shape)
            dwi = dwi.reshape(len(points), grid.size, -1)
            print(dwi.shape)
        return 0

    def track(self):
        SeedBasedTracker.track(self)
        if self.streamlines is not None:
            return
        self._track()
        Cache.get_cache().set(self.id, self.streamlines)
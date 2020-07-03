import torch
import torch.nn as nn
import numpy as np

from dipy.align.imaffine import interpolate_scalar_3d

import dipy.align.imaffine
from src.data import ISMRMDataContainer
from src.data.postprocessing import res100
from src.models import ModelLSTM
from src.tracker import SeedBasedTracker, StreamlinesAlreadyTrackedError
from src.config import Config
from src.util import get_reference_orientation, rotation_from_vectors, get_grid


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

        streamlines = self._track_with_model()
        streamlines = [streamline for streamline in streamlines]
        self.streamlines = streamlines
        self.streamlines = self.filter_streamlines_by_length(streamlines,
                                                             minimum=self.options.min_length,
                                                             maximum=self.options.max_length)
    def _track_with_model(self, no_iterations=200, batch_size=2048):
        model = self.model.cuda()
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        seeds = self.seeds
        reference = get_reference_orientation()

        grid = get_grid(np.array((3, 3, 3))) * self.options.step_width
        streamlines = np.zeros((2 * len(seeds), no_iterations + 1, 3))
        streamlines[:len(seeds), 0] = seeds
        streamlines[len(seeds):, 0] = seeds
        streamline_length = (no_iterations+1) * np.ones([2*len(seeds)], dtype=np.intp)

        batch_size = min(batch_size, len(seeds))
        for idx in range(0, len(seeds), batch_size):
            model.reset()
            idx_range = range(idx, min(idx+batch_size, len(seeds)))
            idx_range_back = range(idx + len(seeds), min(idx+batch_size, len(seeds)) + len(seeds))

            # First prediction
            points = streamlines[idx_range, 0]
            res = self._model_get_next_dir(points, grid)

            res = res/2 # first step is half step in both directions

            streamlines[idx_range, 1] = streamlines[idx_range, 0] + res
            streamlines[idx_range_back, 1] = streamlines[idx_range_back, 0] - res
            rot_mat = np.empty([2*len(res), 3, 3])
            for i, (a, b) in enumerate(model.hidden_state):
                a = torch.repeat_interleave(a, 2, dim=1)
                b = torch.repeat_interleave(b, 2, dim=1)
                model.hidden_state[i] = (a, b)
            #    model.hidden_state.append((a.clone(), b.clone()))
            idx_range = [*idx_range, *idx_range_back]

            for i in range(2, no_iterations + 1):
                dir_vec = streamlines[idx_range, i - 1] - streamlines[idx_range, i - 2]
                for j, vec in enumerate(dir_vec):
                    rotation_from_vectors(rot_mat[j], reference, vec)
                res = self._model_get_next_dir(streamlines[idx_range, i - 1], grid, rot_mat=rot_mat)
                for j, vec in enumerate(dir_vec):
                    theta = np.dot(vec, res[j])
                    if theta < 0:
                        res[j] = -res[j]

                streamlines[idx_range, i] = streamlines[idx_range, i - 1] + res
                valid_points = self._are_voxels_valid(streamlines[idx_range, i])

                for j, is_valid in enumerate(valid_points):
                    if is_valid == 0.:
                        k = idx_range[j]
                        streamline_length[k] = np.min((streamline_length[k], i))
                if np.sum(valid_points) == 0:
                    break
                print("iteration: {}/{}".format(i, no_iterations), end="\r")
            print("{}/{}                                      ".format(idx, len(seeds)))
        streamlines = [sl[:streamline_length[i]] for i, sl in enumerate(streamlines)]
        streamlines = [np.concatenate((s1[:1:-1], s2[1:])) for s1, s2 in
                       zip(streamlines[:len(seeds)], streamlines[len(seeds):])]
        return streamlines

    def _model_get_next_dir(self, points, grid, rot_mat=None):
        if False and rot_mat is not None:
            grid = ((rot_mat.repeat(grid.size/3, axis=0) @
                     grid[None,].repeat(len(points), axis=0).reshape(-1, 3, 1))
                    .reshape((-1, *grid.shape)))
        points_w_grid = points[:, None, None, None, :] + grid
        dwi = self.data_container.get_interpolated_dwi(points_w_grid, ignore_outside_points=True)
        dwi = res100()(dwi, self.data_container.data.b0,
                       self.data_container.data.bvecs,
                       self.data_container.data.bvals)
        dwi = dwi.reshape(len(points), -1)
        dwi = torch.from_numpy(dwi).double().cuda()
        res = self.model(dwi[None, ...]).reshape(-1, 3).cpu().numpy()
        res = res / np.linalg.norm(res, axis=1)[:, None] * self.options.step_width
        #if rot_mat is not None:
        #    res = (rot_mat.swapaxes(1, 2) @ res.reshape(-1, 3, 1)).reshape(-1, 3)
        return res

    def track(self):
        SeedBasedTracker.track(self)
        if self.streamlines is not None:
            return
        self._track()

    def _are_voxels_valid(self, positions, fa_threshold=0.15):
        mask = self.data_container.data.binarymask.astype(float)
        fa = self.data_container.data.fa
        positions = self.data_container.to_ijk(positions)
        a, _ = dipy.align.imaffine.interpolate_scalar_3d(mask, positions)
        b, _ = interpolate_scalar_3d(fa, positions)
        return np.logical_and(np.greater(a, 0), np.greater(b, fa_threshold))

def main():
    data = ISMRMDataContainer()

    model = ModelLSTM(dropout=0.05, hidden_sizes=[256, 256], sizes=(2700, 3),
                      activation_function=nn.Tanh()).double()
    model.load_state_dict(torch.load('model_unrotated_new.pt',
                                     map_location=lambda storage, loc: storage))
    data.get_fa()
    data.normalize().crop()
    tracker = ModelTracker(data, model, seeds_count=30000)
    tracker.track()
    print("noTracked Streamlines:")
    print(len(tracker.streamlines))
    tracker.save_to_file('streamlines_unrotated.vtk')

if __name__ == "__main__":
    main()

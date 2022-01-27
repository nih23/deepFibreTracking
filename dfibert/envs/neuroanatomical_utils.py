from dfibert.data import DataPreprocessor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gym
import numpy as np
import random
import os, sys

sys.path.insert(0, '..')

from collections import deque

from dfibert.tracker.nn.rl import Agent, DQN
import dfibert.envs.RLTractEnvironment as RLTe
from dfibert.tracker import save_streamlines, load_streamlines
from dfibert.envs._state import TractographyState
from tqdm import trange
from dipy.tracking import utils
import dipy.reconst.dti as dti
from dipy.direction import peaks_from_model


def convPoint(p, dims):
    dims = dims - 1
    return (p - dims / 2.) / (dims / 2.)


def interpolate3dAt(data, positions):
    # Warning: data is supposed to be CxHxWxD
    # normalise coordinates into range [-1,1]
    pts = positions.float()
    pts = convPoint(pts, torch.tensor(data.shape[1:4]).to(data.device))
    # reverse pts
    pts = pts[:, (2, 1, 0)]
    # trilinear interpolation
    return torch.nn.functional.grid_sample(data.unsqueeze(0),
                                           pts.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                           align_corners=False, mode="nearest")


class FiberBundleDataset(Dataset):
    def __init__(self, path_to_files, b_val=1000, device="cpu", dataset=None):
        streamlines = load_streamlines(path=path_to_files)

        if dataset is None:
            preprocessor = DataPreprocessor().normalize().crop(b_val).fa_estimate()
            dataset = preprocessor.get_ismrm(f"data/ISMRM2015/")
        self.dataset = dataset
        self.streamlines = [torch.from_numpy(self.dataset.to_ijk(sl)).to(device) for sl in streamlines]
        self.tractMask = torch.zeros(self.dataset.binary_mask.shape)

        for sl in self.streamlines:
            pi = torch.floor(sl).to(torch.long)
            self.tractMask[pi.chunk(chunks=3, dim=1)] = 1

    def __len__(self):
        return len(self.streamlines)

    def __getitem__(self, idx):
        streamline = self.streamlines[idx]
        sl_1 = streamline[0:-2]
        sl_2 = streamline[1:-1]
        return sl_1, sl_2

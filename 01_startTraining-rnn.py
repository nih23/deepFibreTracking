import time
import nrrd
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py

from numpy.random import seed
seed(2342)
from tensorflow import set_random_seed
set_random_seed(4223)
from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.tracking.utils import random_seeds_from_mask
from dipy.reconst.dti import TensorModel
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.segment.mask import median_otsu
from dipy.viz import actor, window
from dipy.io.image import save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core import gradients
from dipy.tracking.streamline import Streamlines
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table, gradient_table_from_bvals_bvecs
from dipy.reconst.dti import fractional_anisotropy
import dipy.reconst.dti as dti
from dipy.tracking import utils

import src.dwi_tools as dwi_tools
import src.nn_helper as nn_helper
import src.tracking as tracking
from src.tied_layers1d import Convolution2D_tied

from dipy.tracking.streamline import values_from_volume
import dipy.align.vector_fields as vfu
from dipy.core.sphere import Sphere
from dipy.core import subdivide_octahedron
from scipy.spatial import KDTree
from dipy.core import subdivide_octahedron

from dipy.tracking.local import LocalTracking
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors
from dipy.tracking.streamline import Streamlines, transform_streamlines
from dipy.tracking import metrics
from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask

import src.rnn_helper as rnn_helper
import numpy as np

import warnings

import tensorflow as tf
from joblib import Parallel, delayed
import multiprocessing
from keras.models import load_model
from keras.layers import Activation

import importlib
import src.tracking as tracking
import tensorflow as tf
from src.nn_helper import swish, squared_cosine_proximity_2
from src.SelectiveDropout import SelectiveDropout
import os
import warnings

import nrrd

def main():
    '''
    train the deep tractography network using previously generated training data
    '''
    # training parameters
    epochs = 100
    lr = 1e-4

    pData = '/data/nico/train/ijk/ismrm_csd_fa015_20mm_noreslicing_mrtrixDenoised_curated/b1000_raw_sw1.0_3x3x3_ut0_rotatedN50_epN0_spc0.50_rotDir1_m001.h5'

    #rnn = rnn_helper.build_streamlineDirectionRNN(units=32)

    noUnits = (32,32)

    rnn = rnn_helper.build_fancyRNN(units=noUnits)

    rnn.summary()

    f = h5py.File(pData, "r")
    train_DWI = np.array(f["train_DWI"].value)
    train_nextDirection = np.array(f["train_NextFibreDirection"].value)
    streamlineIndices = np.array(f["streamlineIndices"].value)
    f.close()

    noMicroBatches = 100
    noStreamlines = len(streamlineIndices)
    start_time = time.time()

    pFile = 'b1000_raw_sw1.0_3x3x3_ut0_rotatedN50_epN0_spc0.50_rotDir1_m001.h5'

    pSave = 'lstm_%s_%d_%d.h5' % (pFile, noUnits[0], noUnits[1])

    print('[INFO] Writing model to: %s' % (pSave))

    for epoch in range(1000):
        acc = []
        for microBatchIndex in range(noMicroBatches):
            k = np.random.randint(noStreamlines-1)
            idxBegin = int(streamlineIndices[k, 0])
            idxEnd = int(streamlineIndices[k + 1, 0])
            trajectory = train_nextDirection[idxBegin:idxEnd, ]
            data = train_DWI[idxBegin:idxEnd,]

            # forward
            for i in range(0, len(trajectory) - 2):
                traj_cur = trajectory[i,]
                traj_cur = traj_cur[np.newaxis, np.newaxis, ...]
                data_cur = data[i,]
                data_cur = data_cur[np.newaxis, ...]
                traj_next = trajectory[i + 1,]
                pStop = np.sum(np.squeeze(traj_next) ** 2) > 0
                traj_next = traj_next[np.newaxis, ...]
                pStop = np.array(pStop[np.newaxis, ...])
                hist = rnn.fit([data_cur], [traj_next, pStop], verbose=0, shuffle=False, batch_size=1)
                acc.append(hist.history['loss'])

            data_cur = data[-1,]
            data_cur = data_cur[np.newaxis, ...]
            traj_next = trajectory[-1,]
            pStop = np.sum(np.squeeze(traj_next) ** 2) > 0
            traj_next = traj_next[np.newaxis, ...]
            pStop = np.array(pStop[np.newaxis, ...])

            hist = rnn.fit([data_cur], [traj_next, pStop], verbose=0, shuffle=False, batch_size=1)
            acc.append(hist.history['loss'])

            rnn.reset_states()

            # backward
            for i in range(1, len(trajectory)-2):
                traj_cur = -1*trajectory[-i,]
                traj_cur = traj_cur[np.newaxis, np.newaxis, ...]
                data_cur = data[i,]
                data_cur = data_cur[np.newaxis, ...]
                traj_next = -1*trajectory[-(i + 1),]
                pStop = np.sum(np.squeeze(traj_next) ** 2) > 0
                traj_next = traj_next[np.newaxis, ...]
                pStop = np.array(pStop[np.newaxis, ...])
                hist = rnn.fit([data_cur], [traj_next, pStop], verbose=0, shuffle=False, batch_size=1)
                acc.append(hist.history['loss'])

            data_cur = data[0,]
            data_cur = data_cur[np.newaxis, ...]
            traj_next = trajectory[-1,]
            pStop = np.sum(np.squeeze(traj_next) ** 2) > 0
            traj_next = traj_next[np.newaxis, ...]
            pStop = np.array(pStop[np.newaxis, ...])
            hist = rnn.fit([data_cur], [traj_next, pStop], verbose=0, shuffle=False, batch_size=1)
            acc.append(hist.history['loss'])


            rnn.reset_states()

        print("[" + str(epoch) + "] " + str(time.time() - start_time) + "s --> " + str(np.mean(acc)))
        rnn.save(pSave)

if __name__ == "__main__":
    main()

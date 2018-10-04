import time
import nrrd
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py

from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
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
from dipy.core import gradients
from dipy.tracking.streamline import Streamlines
from dipy.reconst.dti import fractional_anisotropy

from dipy.tracking import utils

import src.dwi_tools as dwi_tools
import src.nn_helper as nn_helper

from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers import Activation
from keras.callbacks import TensorBoard

pTrainData_fibrePrediction = 'train_prediction_grid_normalized_dti_cs1_wholebrain.h5'
pTrainData_fibreTracking = 'train_tracking_grid_normalized_dti_cs1_wholebrain.h5'
pTrainInput = 'train_input_normalized_dti_cs1_wholebrain_'
noCrossings = 3





def main():
    '''
    train the deep tractography network using previously generated training data
    '''
    #bvals,bvecs,gtab,dwi,aff,t1,binarymask = dwi_tools.loadDataset('100307')
    #dwi_subset, gtab_subset = dwi_tools.cropDatsetToBValue(1000, bvals, bvecs, dwi)
    
    # training parameters
    noGPUs = 1
    batch_size = 2**8
    epochs = 2000
    lr = 3e-3
    useDropout = False
    useBatchNormalization = False
    noFeatures = 2500
    depth = 3
    #activation_function = ReLU()
    activation_function = LeakyReLU()

    
    # load training data
    f = h5py.File(pTrainData_fibrePrediction, "r")
    train_DWI = np.array(f["train_DWI"].value)
    train_prevDirection = np.array(f["train_curPosition"].value)
    train_likelyDirections = np.array(f["train_LikelyFibreDirections"].value)
    train_nextDirection = np.array(f["train_NextFibreDirection"].value)
    f.close()
    
    noSamples,noX,noY,noZ,noD = train_DWI.shape
    
    print('\n**************')
    print('** Training **')
    print('**************\n')
    print('dx ' + str(noX) + ' dy ' + str(noY) + ' dz  ' + str(noZ) + ' dd ' + str(noD))
    print('features ' + str(noFeatures) + ' depth ' + str(depth) + ' lr ' + str(lr) + '\n dropout ' + str(useDropout) + ' bn  ' + str(useBatchNormalization) + ' batch size ' + str(batch_size))
    
    # normalize data
    print('-> projecting dependent value into spherical coordinates')
    train_prevDirection_sph, train_nextDirection_sph = dwi_tools.convertIntoSphericalCoordsAndNormalize(train_prevDirection, train_nextDirection)
    
    print('**************\n')
    
   
    # train simple MLP
    params = "%s_dx_%d_dy_%d_dz_%d_dd_%d_feat_%d_depth_%d_lr_%.3f_dropout_%d_bn_%d" % (activation_function.__class__.__name__,noX,noY,noZ,noD,noFeatures, depth,lr,useDropout,useBatchNormalization)
    pModel = "models/mlp_" + params + "_{epoch:02d}-{val_loss:.6f}.h5"
    checkpoint = ModelCheckpoint(pModel, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    mlp_simple = nn_helper.get_mlp_simpleTracker(lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], outputShape = 2, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)
    mlp_simple.fit([train_DWI], [train_prevDirection_sph,train_nextDirection_sph], batch_size=batch_size, epochs=epochs, verbose=0,validation_split=0.2, callbacks=[checkpoint])
    
    
if __name__ == "__main__":
    main()

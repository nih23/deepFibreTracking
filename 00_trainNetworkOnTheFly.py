from numpy.random import seed
seed(2342)
from tensorflow import set_random_seed
set_random_seed(4223)


import time
import nrrd
import os
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py

import warnings

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
from src.tractographydatagenerator import TractographyDataGenerator

from src.nn_helper import swish

from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers import Activation
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.models import load_model

from dipy.core.gradients import gradient_table, gradient_table_from_bvals_bvecs
from keras.callbacks import LambdaCallback
import argparse

def main():
    '''
    train the deep tractography network using previously generated training data
    '''
    # training parameters
    
    parser = argparse.ArgumentParser(description='Deep Learning Tractography -- Learning @ ISMRM w/ data generator')
    parser.add_argument('streamlines', help='path to streamlines')
    parser.add_argument('-nx', dest='nx', default=1, type=int, help='no of voxels in X plane for each streamline position')
    parser.add_argument('-ny', dest='ny',default=1, type=int, help='no of voxels in Y plane for each streamline position')
    parser.add_argument('-nz', dest='nz',default=1, type=int, help='no of voxels in Z plane for each streamline position')   
    parser.add_argument('-f', '--features', default=128, type=int, help='name of tracking case')
    parser.add_argument('-d', '--depth', default=3, type=int, help='name of tracking case')
    parser.add_argument('-a', '--activationfunction', default='relu', help='relu, leakyrelu, swish')
    parser.add_argument('-m', '--modeltouse', default='mlp_single', help='mlp_single, mlp_doublein_single, cnn_special, cnn_special_pd, rcnn')
    parser.add_argument('-l', '--loss', default='sqCos2', help='cos, mse, sqCos2')
    parser.add_argument('-b', '--batchsize', default=2**12, type=int, help='no. tracking steps')
    parser.add_argument('-e','--epochs', default=1000, type=int, help='no. epochs')
    parser.add_argument('-lr','--learningrate', type=float, default=1e-4, help='minimal length of a streamline [mm]')
    parser.add_argument('-sh', '--shOrder', type=int, default=8, dest='sh', help='order of spherical harmonics (if used)')
    parser.add_argument('--unitTangent', help='unit tangent', dest='unittangent' , action='store_true')
    parser.add_argument('--nounitTangent', help='no unit tangent', dest='unittangent' , action='store_false')
    parser.add_argument('--dropout', help='dropout regularization', dest='dropout' , action='store_true')
    parser.add_argument('--keepZeroVectors', help='keep zero vectors at the outer positions of streamline to indicate termination.', dest='keepzero' , action='store_true')
    parser.add_argument('-bn','--batchnormalization', help='batchnormalization', dest='dropout' , action='store_true')
    parser.add_argument('--bvalue',type=int, default=1000, help='b-value of our DWI data')
        
    parser.set_defaults(unittangent=False)   
    parser.set_defaults(dropout=False)   
    parser.set_defaults(keepzero=False)
    parser.set_defaults(batchnormalization=False)   
    args = parser.parse_args()
    
    noX = args.nx
    noY = args.ny
    noZ = args.nz
    noGPUs = 1
    b_value = args.bvalue
    pStreamlines = args.streamlines
    loss = args.loss
    noFeatures = args.features
    depth = args.depth
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.learningrate
    useDropout = args.dropout
    useBatchNormalization = args.batchnormalization
    usePretrainedModel = False
    unitTangent = args.unittangent
    modelToUse = args.modeltouse
    keepZeroVectors = args.keepzero
    shOrder = args.sh
    
    activation_function = {
          'relu': lambda x: ReLU(),
          'leakyrelu': lambda x: LeakyReLU(),
          'swish': lambda x: Activation(swish)
        }[args.activationfunction](0)
    
    useSphericalCoordinates = False
    pModelOutput = pStreamlines.replace('.vtk','').replace('data/','')
    noOutputNeurons = 3 # euclidean coordinates
    noDiffusionSignals = 100
    
    # load DWI dataset
    nameDWIDataset = 'ISMRM_2015_Tracto_challenge_data'
    useDenoising = False
    bvals,bvecs,gtab,dwi,aff,t1 = dwi_tools.loadISMRMData('data/%s' % (nameDWIDataset), denoiseData = useDenoising, resliceToHCPDimensions=False)
    b0_mask, binarymask = median_otsu(dwi[:,:,:,0], 2, 1)
    nameDWIDataset = 'ISMRM_2015_Tracto_challenge_data_denoised_preproc'
    
    dwi_subset, gtab_subset, bvals_subset, bvecs_subset = dwi_tools.cropDatsetToBValue(b_value, bvals, bvecs, dwi)
    b0_idx = bvals < 10
    b0 = dwi[..., b0_idx].mean(axis=3)
    dwi_singleShell = np.concatenate((dwi_subset, dwi[..., b0_idx]), axis=3)
    #    dwi_singleShell_norm = dwi_tools.normalize_dwi(dwi_singleShell, b0)
    bvals_singleShell = np.concatenate((bvals_subset, bvals[..., b0_idx]), axis=0)
    bvecs_singleShell = np.concatenate((bvecs_subset, bvecs[b0_idx,]), axis=0)
    gtab_singleShell = gradient_table(bvals=bvals_singleShell, bvecs=bvecs_singleShell, b0_threshold = 10)
    t_data, resamplingSphere = dwi_tools.resample_dwi(dwi_subset, b0, bvals_subset, bvecs_subset, sh_order=shOrder, smooth=0, mean_centering=False)
    
    # load streamlines
    streamlines = dwi_tools.loadVTKstreamlines(pStreamlines)
    
    # init data generator
    noTrainingSamples = len(list(range(len(streamlines)-10000)))
    
    training_generator = TractographyDataGenerator(t_data,streamlines,aff,np.array(list(range(len(streamlines)-10000))), dim=[noX,noY,noZ], batch_size=batch_size)
    validation_generator = TractographyDataGenerator(t_data,streamlines,aff,np.array(list(range(len(streamlines)-10000,len(streamlines)))), dim=[noX,noY,noZ], batch_size=batch_size)
    

    
    print('\n**************')
    print('** Training **')
    print('**************\n')
    print('model ' + str(modelToUse) + ' loss ' + loss)
    print('dx ' + str(noX) + ' dy ' + str(noY) + ' dz  ' + str(noZ))
    print('features ' + str(noFeatures) + ' depth ' + str(depth) + ' lr ' + str(lr) + '\ndropout ' + str(useDropout) + ' bn  ' + str(useBatchNormalization) + ' batch size ' + str(batch_size))
    print('dataset ' + str(pStreamlines) + " " + str(len(streamlines)))
    print('**************\n')
    
   
    # train simple MLP
    params = "dg_%s_%s_dx_%d_dy_%d_dz_%d_%s_feat_%d_depth_%d_output_%d_lr_%.4f_dropout_%d_bn_%d_unitTangent_%d_wz_%d" % (modelToUse,loss,noX,noY,noZ,activation_function.__class__.__name__,noFeatures, depth,noOutputNeurons,lr,useDropout,useBatchNormalization,unitTangent,keepZeroVectors)
   
    newpath = r'results/' + pModelOutput + '/models/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
    newpath = r'results/' + pModelOutput + '/logs/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

        ### ### ###
    ### MLP SINGLE ###
        ### ### ###
    mlp_simple = nn_helper.get_mlp_singleOutput(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=[noX,noY,noZ,noDiffusionSignals], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs, normalizeOutput = unitTangent)  
    mlp_simple.summary()

    # Train model on dataset
    pModel = "results/" + pModelOutput + '/models/' + params + "-{val_loss:.6f}.h5"
    pCSVLog = "results/" + pModelOutput + '/logs/' + params + ".csv"
    checkpoint = ModelCheckpoint(pModel, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger(pCSVLog)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-5, verbose=1)
    
    callbacks=[checkpoint, csv_logger]
    
    mlp_simple.fit_generator(generator=training_generator,
        validation_data=validation_generator,
        use_multiprocessing=True,
        workers=6,
        steps_per_epoch=int(noTrainingSamples/batch_size),
        epochs=epochs,
        verbose=1,
        validation_steps=1,
        callbacks=callbacks)
            

        
if __name__ == "__main__":
    main()

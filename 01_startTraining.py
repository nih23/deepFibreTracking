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

from src.nn_helper import swish

from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers import Activation
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.models import load_model

def main():
    '''
    train the deep tractography network using previously generated training data
    '''
    # training parameters
    noGPUs = 1
    batch_size = 2**12
    epochs = 1000
    lr = 1e-4
    useDropout = True
    useBatchNormalization = False 
    usePretrainedModel = False
    unitTension = False
    
    noFeatures = 512
    depth = 3
    activation_function = ReLU()
    #activation_function = LeakyReLU()
    #activation_function = Activation(swish)

    
    # model selection
    modelToUse = 'mlp_doubleIn_single_v4' #'mlp_single' # 'mlp_double'
    modelToUse = 'mlp_doubleIn_single'
    modelToUse = 'mlp_single'
    #modelToUse = 'mlp_single_bitracker'
    #modelToUse = 'mlp_single_bayesian' 
    #modelToUse = 'mlp_doubleIn_single_bayesian'
    modelToUse = 'cnn_special_pd'
    #modelToUse = 'rcnn'
    
    # loss function
    #loss = 'cos'
    #loss = 'mse'
    loss = 'sqCos2'
    
    useSphericalCoordinates = False

    # whole brain tractography
    
    if( (modelToUse == 'cnn_special')   or  (modelToUse == 'rcnn')):
        #pTrainData = 'data/train_resampled_10x10_noB0InSH_step0.6_wholeBrain_b1000_2_ukf_curated.h5'
        pTrainData = 'data/train_res1002D_16x16_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension.h5'
        depth = 3
        noFeatures = 32
    elif( (modelToUse == 'cnn_special_pd')   or  (modelToUse == 'rcnn_pd')):
        pTrainData = 'data/train_res1002D_16x16_prevDWI_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension.h5'
        depth = 3
        noFeatures = 32
        
        f = h5py.File(pTrainData, "r")
        train_DWI_past = np.array(f["train_DWI_prev"].value)
        f.close()
    else:
        #### GENERATE UKF b1k TRAINING DATA AND RUN TRAINING ON MLP v3 and v1
        #pTrainData = 'data/train_sh8_noB0InSH_step0.6_wholeBrain_b1000_2_ukf_curated_1x1x1.h5'
        pTrainData = 'data/train_sh8_noB0InSH_step0.6_wholeBrain_b1000_2_ukf_curated_1x1x1_noUnitTension.h5'
        #pTrainData = 'data/train_sh8_noB0InSH_step0.6_wholeBrain_b1000_2_csd_ismrm_1x1x1.h5'
        #pTrainData = 'data/train_sh8_noB0InSH_step0.6_wholeBrain_b1000_2_csd_ismrm_1x1x1_noUnitTension.h5'
        
        pTrainData = 'data/train_res100_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension.h5'
        pTrainData = 'data/train_res1002D_16x16_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension.h5'

    pModelOutput = pTrainData.replace('.h5','').replace('data/','')

    
    if(useSphericalCoordinates == True):
        noOutputNeurons = 2 # spherical coordinates
    else:
        noOutputNeurons = 3 # euclidean coordinates
    
    # load training data
    f = h5py.File(pTrainData, "r")
    train_DWI = np.array(f["train_DWI"].value)
    train_prevDirection = np.array(f["train_curPosition"].value)
    #train_likelyDirections = np.array(f["train_LikelyFibreDirections"].value)
    train_nextDirection = np.array(f["train_NextFibreDirection"].value)
    f.close()
    
    #train_prevDirection = (train_prevDirection + 1) / 2
    #train_nextDirection = (train_nextDirection + 1) / 2

    # remove streamline points right at the end of the streamline
    # zvFix didnt account for zero vectors in the previous direction
    # zvFix2 removes zero vectors in both previous and next streamline directions
    vN = np.sqrt(np.sum(train_nextDirection ** 2 , axis = 1))
    idx1 = np.where(vN > 0)[0]
    vN = np.sqrt(np.sum(train_prevDirection ** 2 , axis = 1))
    idx2 = np.where(vN > 0)[0]
    s2 = set(idx2)
    idxNoZeroVectors = [val for val in idx1 if val in s2]
    
    #train_DWI = train_DWI[idxNoZeroVectors,...]
    #train_nextDirection = train_nextDirection[idxNoZeroVectors,]
    #train_prevDirection = train_prevDirection[idxNoZeroVectors,]
    
    noX = 1
    noY = 1
    noZ = 1
    noD = 1
    
    noSamples,noX,noY,noZ,noD = train_DWI.shape
    
    print('\n**************')
    print('** Training **')
    print('**************\n')
    print('model ' + str(modelToUse) + ' loss ' + loss)
    print('dx ' + str(noX) + ' dy ' + str(noY) + ' dz  ' + str(noZ) + ' dd ' + str(noD))
    print('features ' + str(noFeatures) + ' depth ' + str(depth) + ' lr ' + str(lr) + '\ndropout ' + str(useDropout) + ' bn  ' + str(useBatchNormalization) + ' batch size ' + str(batch_size))
    print('dataset ' + str(pTrainData))
    print('**************\n')
    
   
    # train simple MLP
    params = "%s_%s_dx_%d_dy_%d_dz_%d_dd_%d_%s_feat_%d_depth_%d_output_%d_lr_%.4f_dropout_%d_bn_%d_pt_%d_unitTension_%d-zvFix2" % (modelToUse,loss,noX,noY,noZ,noD,activation_function.__class__.__name__,noFeatures, depth,noOutputNeurons,lr,useDropout,useBatchNormalization,usePretrainedModel,unitTension)
    pModel = "results/" + pModelOutput + '/models/V3wZ_' + params + "_{epoch:02d}-{val_loss:.6f}.h5"
    pCSVLog = "results/" + pModelOutput + '/logs/V3wZ_' + params + ".csv"
    
    newpath = r'results/' + pModelOutput + '/models/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
    newpath = r'results/' + pModelOutput + '/logs/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    if(noOutputNeurons == 2):
        # spherical coordinates
        warnings.warn('conversion into spherical coordinates seems to be flawed atm')
        
        print('-> projecting dependent value into spherical coordinates')
        train_prevDirection, train_nextDirection = dwi_tools.convertIntoSphericalCoordsAndNormalize(train_prevDirection, train_nextDirection)    
    
    if (usePretrainedModel):
        checkpoint = ModelCheckpoint(pModel, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        csv_logger = CSVLogger(pCSVLog)
        ptmodel = load_model(pPretrainedModel)
        ptmodel.fit([train_DWI], [train_prevDirection, train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
        return
    
    checkpoint = ModelCheckpoint(pModel, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger(pCSVLog)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-5, verbose=1)

   
    ### train model ###
    if (modelToUse == 'mlp_double'):
        mlp_simple = nn_helper.get_mlp_doubleOutput(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)       
        mlp_simple.fit([train_DWI], [train_prevDirection, train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ###
    elif (modelToUse == 'cnn_special_pd'):
        print(str(train_DWI.shape))
        train_DWI = np.reshape(train_DWI, [noSamples,16,16])
        print(str(train_DWI.shape))

        #train_DWI_past = train_DWI_past[idxNoZeroVectors,...]
        train_DWI_past = np.reshape(train_DWI_past, [noSamples,16,16])
        
        train_DWI_both = np.stack((train_DWI_past, train_DWI))
        train_DWI_both = np.moveaxis(train_DWI_both,0,-1)
        print(str(train_DWI_both.shape))
        
        cnn = nn_helper.get_cnn_singleOutput(loss=loss, lr=lr, useBN = useBatchNormalization, inputShapeDWI=train_DWI_both.shape[1:], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)     
        cnn.summary()
        cnn.fit([train_DWI_both], [train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
        
    elif (modelToUse == 'cnn_special'):
        print(str(train_DWI.shape))
        train_DWI = np.reshape(train_DWI, [noSamples,16,16])
        train_DWI = train_DWI[..., np.newaxis]
        print(str(train_DWI.shape))
        cnn = nn_helper.get_cnn_singleOutput(loss=loss, lr=lr, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)     
        cnn.summary()
        cnn.fit([train_DWI], [train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    
    ###
    ###
    
    elif (modelToUse == 'rcnn'):
        print(str(train_DWI.shape))
        train_DWI = np.reshape(train_DWI, [noSamples,16,16])
        train_DWI = train_DWI[..., np.newaxis]
        print(str(train_DWI.shape))
        cnn = nn_helper.get_rcnn(loss=loss, lr=lr, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)     
        cnn.summary()
        cnn.fit([train_DWI], [train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    
    ###
    
    
    elif (modelToUse == 'mlp_single_bayesian'):
        mlp_simple = nn_helper.get_mlp_singleOutput_bayesian(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)       
        mlp_simple.fit([train_DWI], [train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ###
    
    
    elif (modelToUse == 'mlp_doubleIn_single_bayesian'):
        ### mirror input/output to enable tracking in two directions ###
        tensionInput = np.concatenate( (train_prevDirection, train_nextDirection) )
        tensionOutput = np.concatenate( (train_nextDirection, train_prevDirection) )
        train_DWI_repeated = np.concatenate((train_DWI,train_DWI) )
        ### fire
        mlp_simple = nn_helper.get_mlp_multiInput_singleOutput_bayesian(loss=loss, lr=lr, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], inputShapeVector=(3,), outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)       
        mlp_simple.fit([train_DWI_repeated, tensionInput], [tensionOutput], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
        #mlp_simple.fit([train_DWI, train_prevDirection], [train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ###
    
    
    elif (modelToUse == 'mlp_doubleIn_single'):
        ### mirror input/output to enable tracking in two directions ###
        #tensionInput = np.concatenate( (train_prevDirection, train_nextDirection) )
        #tensionOutput = np.concatenate( (train_nextDirection, train_prevDirection) )
        #train_DWI_repeated = np.concatenate((train_DWI,train_DWI) )
        ### fire
        mlp_simple = nn_helper.get_mlp_multiInput_singleOutput(loss=loss, lr=lr, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], inputShapeVector=(3,), outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs, normalizeOutput = unitTension)       
        #mlp_simple.fit([train_DWI_repeated, tensionInput], [tensionOutput], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
        
        #V1
        #mlp_simple.fit([train_DWI, train_prevDirection], [train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
        
        #V2 
        #mlp_simple.fit([train_DWI, -1*train_prevDirection], [train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
        
        #V3
        train_DWI_repeated = np.concatenate((train_DWI,train_DWI) )
        tensionInput = np.concatenate( (-1*train_prevDirection, train_nextDirection) )
        tensionOutput = np.concatenate( (train_nextDirection, -1*train_prevDirection) )
        mlp_simple.fit([train_DWI_repeated, tensionInput], [tensionOutput], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
        
    elif (modelToUse == 'mlp_doubleIn_single_v4'):
        ### fire
        mlp_simple = nn_helper.get_mlp_multiInput_singleOutput_v4(loss=loss, lr=lr, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], inputShapeVector=(3,), outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs, normalizeOutput = unitTension)       
        
        #V3
        train_DWI_repeated = np.concatenate((train_DWI,train_DWI) )
        tensionInput = np.concatenate( (-1*train_prevDirection, train_nextDirection) )
        tensionOutput = np.concatenate( (train_nextDirection, -1*train_prevDirection) )
        mlp_simple.fit([train_DWI_repeated, tensionInput], [tensionOutput], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
        
    elif (modelToUse == 'mlp_doubleIn_single_v2'):
        ### mirror input/output to enable tracking in two directions ###
        tensionInput = np.concatenate( (train_prevDirection, train_nextDirection) )
        tensionOutput = np.concatenate( (train_nextDirection, train_prevDirection) )
        train_DWI_repeated = np.concatenate((train_DWI,train_DWI) )
        ### fire
        mlp_simple = nn_helper.get_mlp_multiInput_singleOutput_v2(loss=loss, lr=lr, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], inputShapeVector=(3,), outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)       
        mlp_simple.fit([train_DWI_repeated, tensionInput], [tensionOutput, tensionOutput], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
        
        
    elif (modelToUse == 'mlp_doubleIn_single_v3'):
        ### mirror input/output to enable tracking in two directions ###
        tensionInput = np.concatenate( (train_prevDirection, train_nextDirection) )
        tensionOutput = np.concatenate( (train_nextDirection, train_prevDirection) )
        train_DWI_repeated = np.concatenate((train_DWI,train_DWI) )
        szTO = tensionOutput.shape
        ### fire
        mlp_simple = nn_helper.get_mlp_multiInput_singleOutput_v3(loss=loss, lr=lr, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], inputShapeVector=(3,), outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)       
        mlp_simple.fit([train_DWI_repeated, tensionInput], [tensionOutput, np.zeros([szTO[0],3])], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
        
        
    elif (modelToUse == 'mlp_single'):
        mlp_simple = nn_helper.get_mlp_singleOutput(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs, normalizeOutput = unitTension)       
        mlp_simple.fit([train_DWI], [train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ###
    
    elif (modelToUse == 'mlp_single_2'):
        # this model doesnt make sense ...
        train_DWI_repeated = np.concatenate((train_DWI,train_DWI) )
        tensionOutput = np.concatenate( (train_nextDirection, -1*train_prevDirection) )
        mlp_simple = nn_helper.get_mlp_singleOutput(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs, normalizeOutput = unitTension)       
        mlp_simple.fit([train_DWI_repeated], [tensionOutput], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ###
    
    elif (modelToUse == 'mlp_single_bitracker'):
        mlp_simple = nn_helper.get_mlp_singleInput_doubleOutput(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)       
        mlp_simple.fit([train_DWI], [train_prevDirection, train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ###
    
    
    elif (modelToUse == 'unet'):
        cnn_simple = nn_helper.get_3Dunet_simpleTracker(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)       
        cnn_simple.fit([train_DWI], [train_prevDirection, train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    
    
    
    elif (modelToUse == 'endTracking_doubleIn_single'):
        idxNotracking = np.where(  (train_nextDirection == [0,0,0]).all(axis=1)  )[0]
        stopTracking = np.zeros(len(train_nextDirection))
        stopTracking[idxNotracking] = 1
        
        endTracking = nn_helper.get_mlp_multiInput_detectEndingStreamlines(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], features = noFeatures, depth = depth, noGPUs=noGPUs)       
        endTracking.fit([train_DWI, train_prevDirection], [stopTracking], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ###
    
    
if __name__ == "__main__":
    main()

from keras.layers import Dense, Activation, Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda, Flatten, BatchNormalization, UpSampling2D, LeakyReLU, PReLU, Dropout, AveragePooling1D, Reshape, Permute, Add, ELU, Conv3D, MaxPooling3D, UpSampling3D, Conv2D, MaxPooling2D, Multiply, LSTM, CuDNNLSTM, ReLU
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.models import Model, load_model
from keras.constraints import nonneg
from keras import optimizers, losses
from keras import backend as K
from keras.utils import multi_gpu_model
import keras

from src.tied_layers1d import Convolution2D_tied
from src.SelectiveDropout import SelectiveDropout
import src.dwi_tools as dwi_tools
import sys, getopt
import tensorflow as tf
import h5py
import numpy as np
import time
from keras import backend as K
from keras.layers.merge import add

import time
import nrrd
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py

from dipy.tracking import eudx
from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.tracking.utils import random_seeds_from_mask
from dipy.reconst.dti import TensorModel, quantize_evecs
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model, DeterministicMaximumDirectionGetter
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.segment.mask import median_otsu
from dipy.viz import actor, window
from dipy.io.image import save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core import gradients
from dipy.tracking.streamline import Streamlines, transform_streamlines
from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table, gradient_table_from_bvals_bvecs
from dipy.reconst.dti import fractional_anisotropy
from dipy.tracking import utils

import src.dwi_tools as dwi_tools
import src.nn_helper as nn_helper

from src.nn_helper import squared_cosine_proximity_2
from dipy.segment.mask import median_otsu

import random


def squared_cosine_proximity_WEP(y_true, y_pred):
    '''
    squares cosine loss function (variant 2)
    This loss function allows the network to be invariant wrt. to the streamline orientation. The direction of a vector v_i (forward OR backward (-v_i)) doesn't affect the loss.
    '''
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    # ||y_gt||_2 * cos^2(y_gt,y_est) - (||y_gt||_2 - 1) * ||y_pred||_2
    return - tf.multiply( (K.sum(y_true**2, axis=1)), (K.sum(y_true * y_pred, axis=-1) ** 2) ) - tf.multiply( (K.sum(y_true**2, axis=1) - 1), (K.sum(y_pred**2, axis=1) - 1))


def build_model(inputShapeDWI=[1,100], features=512, units=10, useDropout=True, activation_function=ReLU(),lr=1e-4,decayrate=0):

    pDropout = 0.0
    
    if(useDropout):
        pDropout = 0.2
    
    inputs = Input(inputShapeDWI, batch_shape=(1,1,100))
    layers = [inputs]
    
    ####
    # predicting next direction stream
    ####
    layers.append(CuDNNLSTM(stateful = True, return_state=False, return_sequences=True, units=units)(layers[0]))
    layers.append(CuDNNLSTM(stateful = True, return_state=False, return_sequences=False, units=units)(layers[-1]))
#    layers.append(STM(stateful = True, return_state=False, return_sequences=True, units=units,dropout=pDropout, recurrent_dropout=pDropout, implementation=2)(layers[0]))
#    layers.append(LSTM(stateful = True, return_state=True, return_sequences=False, units=units,dropout=pDropout, recurrent_dropout=pDropout, implementation=2)(layers[-1]))

    layers.append(concatenate(layers[-1], axis = -1))
    #layers.append(Flatten()(layers[-1]))
    layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Dense(3, kernel_initializer = 'he_normal')(layers[-1]))
    dirLayer = layers[-1]
    
    ####
    # terminate tracking stream
    ####
    layers.append(CuDNNLSTM(stateful = True, return_state=False, return_sequences=True, units=units)(layers[0]))
    layers.append(CuDNNLSTM(stateful = True, return_state=False, return_sequences=False, units=units)(layers[-1]))   
#    layers.append(LSTM(stateful = True, return_state=False, return_sequences=True, units=units,dropout=pDropout, recurrent_dropout=pDropout, implementation=2)(layers[0]))
#    layers.append(LSTM(stateful = True, return_state=True, return_sequences=False, units=units,dropout=pDropout, recurrent_dropout=pDropout, implementation=2)(layers[-1]))
    
    layers.append(concatenate(layers[-1], axis = -1))
    #layers.append(Flatten()(layers[-1]))
    layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(activation_function(layers[-1]))
    #layers.append(Flatten()(layers[-1]))
    layers.append(Dense(1, kernel_initializer = 'he_normal', activation='sigmoid', name = 'signLayer')(layers[-1]))
    
    outputSignLayer = layers[-1]
    layers.append(concatenate(  [layers[-1], layers[-1], layers[-1]], axis = -1))
    signLayerConcat = layers[-1]
    
    # join both streams
    layers.append(Multiply()([dirLayer,signLayerConcat]))
    outputPredictedTangent = layers[-1]
    
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    rnn = Model((layers[0]), outputs=(outputPredictedTangent,outputSignLayer))
    rnn.compile(loss=[squared_cosine_proximity_2, losses.binary_crossentropy], optimizer=optimizer)
    return rnn


def build_streamlineDirectionRNN(units=128, useDropout=True, activation_function=ReLU(), lr=1e-4, decayrate=0):
    pDropout = 0.0

    if (useDropout):
        pDropout = 0.2

    inputs = Input((1,3), batch_shape=(1,1,3))
    layers = [inputs]

    ####
    # predicting next direction stream
    ####
    #layers.append(CuDNNLSTM(stateful=True, return_state=False, return_sequences=True, units=units)(layers[0])) # maybe use return sequence and feed that into another CNN or GAP layer
    #layers.append(CuDNNLSTM(stateful=True, return_state=False, return_sequences=False, units=units)(layers[-1]))
    #layers.append(LSTM(stateful=True, return_state=False, return_sequences=False, units=units, recurrent_dropout = 0.3, dropout = 0.3)(layers[-1]))
    layers.append(Dense(3, kernel_initializer='he_normal', activation='tanh', name='tangentPrediction')(layers[-1]))
    #outputPredictedTangent = layers[-1]

    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    rnn = Model((layers[0]), outputs=(layers[-1]))
    rnn.compile(loss=[squared_cosine_proximity_2], optimizer=optimizer)
    return rnn



def build_mightyRNN(units=(32,128), useDropout=True, activation_function=ReLU(), lr=1e-4, decayrate=0, inputShapeDWI = (3,3,3,32)):
    pDropout = 0.0

    if (useDropout):
        pDropout = 0.2

    i1 = Input((1,3), batch_shape=(1,1,3))

    i2 = Input((1, 3), batch_shape= (1,) + inputShapeDWI)

    layers = [i1]
    layers.append(LSTM(stateful=True, return_state=False, return_sequences=True, units=units[0], recurrent_dropout = 0.3, dropout = 0.3)(
        layers[0]))  # maybe use return sequence and feed that into another CNN or GAP layer
    layLSTMdirection = layers[-1]
    layers.append([i2])
    layI2 = layers[-1]
    layers.append(LSTM(stateful=True, return_state=False, return_sequences=True, units=units[1], recurrent_dropout = 0.3, dropout = 0.3)(
        layers[0]))  # maybe use return sequence and feed that into another CNN or GAP layer
    layLSTMdwi = layers[-1]

    ####
    # predict next direction stream
    ####
    layers.append(concatenate([layLSTMdirection,layLSTMdwi]))
    layers.append(LSTM(stateful=True, return_state=False, return_sequences=False, units=units[1], recurrent_dropout = 0.3, dropout = 0.3)(layers[-1]))
    layers.append(Dense(3, kernel_initializer='he_normal', activation='tanh', name='tangentPrediction')(layers[-1]))

    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    rnn = Model((i1, i2), outputs=(layers[-1]))
    rnn.compile(loss=[squared_cosine_proximity_2], optimizer=optimizer)
    return rnn


def build_fancyRNN(units=(32,128), useDropout=True, activation_function=ReLU(), lr=1e-4, decayrate=0, inputShapeDWI = (3,3,3,32)):
    pDropout = 0.0

    if (useDropout):
        pDropout = 0.2

    #i1 = Input((1, 3), batch_shape= (1,1,np.prod(inputShapeDWI)))
    i1 = Input((1, 3), batch_shape=(1,) + inputShapeDWI)

    layers = [i1]

    layers.append(Reshape((-1,np.prod(inputShapeDWI)))(layers[-1]))

    layers.append(LSTM(stateful=True, return_state=False, return_sequences=True, units=units[0], input_dim=inputShapeDWI, recurrent_dropout = 0.3, dropout = 0.3)(
        layers[-1]))  # maybe use return sequence and feed that into another CNN or GAP layer

    ####
    # predict next direction stream
    ####
    #layers.append(LSTM(stateful=True, return_state=False, return_sequences=False, units=units[1], recurrent_dropout = 0.3, dropout = 0.3)(layers[-1]))
    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(3, kernel_initializer='he_normal', activation='tanh', name='tangentPrediction')(layers[-1]))
    o_tangent = layers[-1]

    ####
    # predict next direction stream
    ####
    layers.append(Dense(1, kernel_initializer='he_normal', activation='sigmoid', name='stoppingProb')(layers[-1]))
    o_pStop = layers[-1]
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    rnn = Model((i1), outputs=(o_tangent, o_pStop))
    rnn.compile(loss=[squared_cosine_proximity_WEP, losses.binary_crossentropy], optimizer=optimizer)
    return rnn


def train(pStreamlineData, noEpochs = 50, lr = 1e-4, batchSize = 2**9, features = 128, units = 20):
    model = build_model(lr=lr, features=features, units = units)
    model.summary()
    # load training data
    pCaseID = 'ISMRM_2015_Tracto_challenge_data'
    useDenoising = False
    b_value = 1000
    bvals,bvecs,gtab,dwi,aff,t1 = dwi_tools.loadISMRMData('data/%s' % (pCaseID), denoiseData = useDenoising, resliceToHCPDimensions=False)
    b0_mask, binarymask = median_otsu(dwi[:,:,:,0], 2, 1)
    # crop DWI data
    dwi_subset, gtab_subset, bvals_subset, bvecs_subset = dwi_tools.cropDatsetToBValue(b_value, bvals, bvecs, dwi)
    b0_idx = bvals < 10
    b0 = dwi[..., b0_idx].mean(axis=3)
    dwi_singleShell = np.concatenate((dwi_subset, dwi[..., b0_idx]), axis=3)
    #    dwi_singleShell_norm = dwi_tools.normalize_dwi(dwi_singleShell, b0)
    bvals_singleShell = np.concatenate((bvals_subset, bvals[..., b0_idx]), axis=0)
    bvecs_singleShell = np.concatenate((bvecs_subset, bvecs[b0_idx,]), axis=0)
    gtab_singleShell = gradient_table(bvals=bvals_singleShell, bvecs=bvecs_singleShell, b0_threshold = 10)
    dwi_subset_100, resamplingSphere = dwi_tools.resample_dwi(dwi_subset, b0, bvals_subset, bvecs_subset, sh_order=8, smooth=0, mean_centering=False)
    
    # save path
    pSave = "rnn_f%d_u%d" % (features, units)
    
    # load streamlines
    streamlines = dwi_tools.loadVTKstreamlines(pStreamlineData)
    random.shuffle(streamlines)
    noTrainSamples = 1000
    noTestSamples = 100

    sl_train = streamlines[0:noTrainSamples]
    sl_test = streamlines[noTrainSamples:noTrainSamples+noTestSamples]
    
    # start training
    for epoch in range(0,noEpochs):
        sl_train_1k = np.random.choice(sl_train,1000, replace=False)
        sl_test_100 = np.random.choice(sl_test,100, replace=False)
        noStreamlines = len(sl_train)
        interpolatedDWISubvolume, directionToPreviousStreamlinePoint, directionToNextStreamlinePoint, interpolatedDWISubvolumePast,slOffset,_ = dwi_tools.generateTrainingData(sl_train_1k, dwi_subset_100, unitTension = False, affine=aff, step = 0.6)

        interpolatedDWISubvolume_test, _, directionToNextStreamlinePoint_test, _,slOffset_test,_ = dwi_tools.generateTrainingData(sl_test_100, dwi_subset_100, unitTension = False, affine=aff, step = 0.6)
    

        
        # train model
        mean_tr_loss = []
        mean_tr_loss1 = []
        mean_tr_loss2 = []
        for i in range(0,len(sl_train)):
            noSamples = len(sl_train[i])
            # build weight matrix
            class_weight = { 'signLayer': {0: noSamples-1 , 1: 1 } }

            # interpolate dwi data
            dwiCoefficients_i = interpolatedDWISubvolume[int(slOffset[i]):int(slOffset[i])+noSamples,]
            tangent_i = directionToNextStreamlinePoint[int(slOffset[i]):int(slOffset[i])+noSamples,]
            # reshape (plug into sequence generator)
            dwiCoefficients_i = dwiCoefficients_i.reshape([noSamples,1,-1])
            labels_i = np.ones((noSamples,1))
            labels_i[-1] = 0
            # fit lstm
            hist = model.fit([dwiCoefficients_i],[tangent_i, labels_i], verbose=0, class_weight=class_weight, shuffle=False, batch_size=1)
            mean_tr_loss.append(hist.history['loss'])
            mean_tr_loss1.append(hist.history['multiply_1_loss'])
            mean_tr_loss2.append(hist.history['signLayer_loss'])
            if((i % 100) == 0):
                print('Train ' + str(i)+'@'+str(epoch) + ' -> ' + str(np.mean(mean_tr_loss)) + ';' + str(np.mean(mean_tr_loss1)) + ';' + str(np.mean(mean_tr_loss2)))
            # reset_states
            model.reset_states()      
        print('Train final ' + str(i)+'@'+str(epoch) + ' -> ' + str(np.mean(mean_tr_loss)) + ';' + str(np.mean(mean_tr_loss1)) + ';' + str(np.mean(mean_tr_loss2)))
        
        # test model
        mean_tr_loss = []
        mean_tr_loss1 = []
        mean_tr_loss2 = []
        for i in range(0,len(sl_test)):
            noSamples = len(sl_test[i])
            # interpolate dwi data
            dwiCoefficients_i = interpolatedDWISubvolume_test[int(slOffset_test[i]):int(slOffset_test[i])+noSamples,]
            tangent_i = directionToNextStreamlinePoint_test[int(slOffset_test[i]):int(slOffset_test[i])+noSamples,]
            # reshape (plug into sequence generator)
            dwiCoefficients_i = dwiCoefficients_i.reshape([noSamples,1,-1])
            labels_i = np.ones((noSamples,1))
            labels_i[-1] = 0
            # fit lstm
            loss,mult_loss,sign_loss = model.evaluate([dwiCoefficients_i],[tangent_i, labels_i], verbose=0, batch_size=1)
            mean_tr_loss.append(loss)
            mean_tr_loss1.append(mult_loss)
            mean_tr_loss2.append(sign_loss)
            # reset_states
            model.reset_states()
        
        print('Test final ' + str(i)+'@'+str(epoch) + ' -> ' + str(np.mean(mean_tr_loss)) + ';' + str(np.mean(mean_tr_loss1)) + ';' + str(np.mean(mean_tr_loss2)))
        pSaveIntRes = "%s_e%d_l%.5f.h5" % (pSave, epoch, np.mean(mean_tr_loss))
        model.save(pSaveIntRes)
        print('Stored model: ' + pSaveIntRes)
        
    return model

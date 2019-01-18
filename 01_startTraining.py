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

from src.nn_helper import swish

from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers import Activation
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.models import load_model

import argparse

from imblearn.keras import BalancedBatchGenerator

def main():
    '''
    train the deep tractography network using previously generated training data
    '''
    # training parameters
    
    parser = argparse.ArgumentParser(description='Deep Learning Tractography -- Learning')
    parser.add_argument('data', help='path to training data')
    parser.add_argument('-f', '--features', default=128, type=int, help='name of tracking case')
    parser.add_argument('-d', '--depth', default=3, type=int, help='name of tracking case')
    parser.add_argument('-a', '--activationfunction', default='relu', help='relu, leakyrelu, swish')
    parser.add_argument('-m', '--modeltouse', default='mlp_single', help='mlp_single, mlp_doublein_single, cnn_special, cnn_special_pd, rcnn')
    parser.add_argument('-l', '--loss', default='sqCos2WEP', help='cos, mse, sqCos2, sqCos2WEP')
    parser.add_argument('-b', '--batchsize', default=2**12, type=int, help='no. tracking steps')
    parser.add_argument('-e','--epochs', default=1000, type=int, help='no. epochs')
    parser.add_argument('-lr','--learningrate', type=float, default=1e-4, help='minimal length of a streamline [mm]')
    parser.add_argument('-sh', '--shOrder', type=int, default=8, help='order of spherical harmonics (if used)')
    parser.add_argument('--unitTangent', help='unit tangent', dest='unittangent' , action='store_true')
    parser.add_argument('--dropout', help='dropout regularization', dest='dropout' , action='store_true')
    parser.add_argument('--keepZeroVectors', help='keep zero vectors at the outer positions of streamline to indicate termination.', dest='keepzero' , action='store_true')
    parser.add_argument('-bn','--batchnormalization', help='batchnormalization', dest='dropout' , action='store_true')
    parser.add_argument('--bvalue',type=int, default=1000, help='b-value of our DWI data')
        
    parser.set_defaults(unittangent=False)   
    parser.set_defaults(dropout=False)   
    parser.set_defaults(keepzero=False)
    parser.set_defaults(batchnormalization=False)   
    args = parser.parse_args()
    
    noGPUs = 1
    pTrainData = args.data
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
    activation_function = {
          'relu': lambda x: ReLU(),
          'leakyrelu': lambda x: LeakyReLU(),
          'swish': lambda x: Activation(swish)
        }[args.activationfunction](0)
    
    if(loss == 'sqCos2WEP'):
        print('Keeping zero vectors as the neural network shall predict endpoints.')
        keepZeroVectors = True
    
    useSphericalCoordinates = False
    pModelOutput = pTrainData.replace('.h5','').replace('data/','')

    
    if(useSphericalCoordinates == True):
        noOutputNeurons = 2 # spherical coordinates
    else:
        noOutputNeurons = 3 # euclidean coordinates
    
    # load training data
    f = h5py.File(pTrainData, "r")
    train_DWI = np.array(f["train_DWI"].value)
    train_prevDirection = np.array(f["train_curPosition"].value)
    train_nextDirection = np.array(f["train_NextFibreDirection"].value)
    f.close()
    
    indices = np.arange(len(train_DWI))
    np.random.shuffle(indices)
    train_DWI = train_DWI[indices,]
    train_nextDirection = train_nextDirection[indices,]
    train_prevDirection = train_prevDirection[indices,]
       
    vN = np.sqrt(np.sum(train_nextDirection ** 2 , axis = 1))
    idx1 = np.where(vN > 0)[0]
    vN = np.sqrt(np.sum(train_prevDirection ** 2 , axis = 1))
    idx2 = np.where(vN > 0)[0]
    s2 = set(idx2)
    idxNoZeroVectors = [val for val in idx1 if val in s2]
    
    if(keepZeroVectors == False):
        train_DWI = train_DWI[idxNoZeroVectors,...]
        train_nextDirection = train_nextDirection[idxNoZeroVectors,]
        train_prevDirection = train_prevDirection[idxNoZeroVectors,]
    
    noSamples,noX,noY,noZ,noD = train_DWI.shape
    
    print('\n**************')
    print('** Training **')
    print('**************\n')
    print('model ' + str(modelToUse) + ' loss ' + loss)
    print('dx ' + str(noX) + ' dy ' + str(noY) + ' dz  ' + str(noZ) + ' dd ' + str(noD))
    print('features ' + str(noFeatures) + ' depth ' + str(depth) + ' lr ' + str(lr) + '\ndropout ' + str(useDropout) + ' bn  ' + str(useBatchNormalization) + ' batch size ' + str(batch_size))
    print('dataset ' + str(pTrainData) + " " + str(noSamples))
    print('**************\n')
    
   
    # train simple MLP
    params = "%s_%s_dx_%d_dy_%d_dz_%d_dd_%d_%s_feat_%d_depth_%d_output_%d_lr_%.4f_dropout_%d_bn_%d_unitTangent_%d_wz_%d" % (modelToUse,loss,noX,noY,noZ,noD,activation_function.__class__.__name__,noFeatures, depth,noOutputNeurons,lr,useDropout,useBatchNormalization,unitTangent,keepZeroVectors)
    pModel = "results/" + pModelOutput + '/models/' + params + "-{val_loss:.6f}.h5"
    pCSVLog = "results/" + pModelOutput + '/logs/' + params + ".csv"
    
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

       ####################
    ####################
    if(modelToUse == '2Dcnn'):
        noX = 1
        noY = 1
        noZ = 1
        modelToUse = '2D_cnn'
        model = nn_helper.get_simpleCNN(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=[8,8,1], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)  
        model.summary()
        # data generator
        training_generator = TwoDimensionalTractographyDataGenerator(t_data,streamlines,aff,np.array(list(range(len(streamlines)-20000))), dim=[noX,noY,noZ], batch_size=batch_size, storeTemporaryData = storeTemporaryData)
        validation_generator = TwoDimensionalTractographyDataGenerator(t_data,streamlines,aff,np.array(list(range(len(streamlines)-20000,len(streamlines)-10000))), dim=[noX,noY,noZ], batch_size=batch_size, storeTemporaryData = storeTemporaryData)
    ####################
    ####################
    if(modelToUse == '3Dcnn'):
        modelToUse = '3D_cnn_dr%s' % (str(dilationRate))
        model = nn_helper.get_simple3DCNN(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=[noX*8,noY*8,noZ,1], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs, dilationRate = dilationRate)  
        model.summary()
        # data generator
        training_generator = ThreeDimensionalTractographyDataGenerator(t_data,streamlines,aff,np.array(list(range(len(streamlines)-10000))), dim=[noX,noY,noZ], batch_size=batch_size, storeTemporaryData = storeTemporaryData)
        validation_generator = ThreeDimensionalTractographyDataGenerator(t_data,streamlines,aff,np.array(list(range(len(streamlines)-10000,len(streamlines)))), dim=[noX,noY,noZ], batch_size=batch_size, storeTemporaryData = storeTemporaryData)
    ####################
    ####################
    ### train model ###
    if (modelToUse == 'mlp_double'):
        mlp_simple = nn_helper.get_mlp_doubleOutput(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)       
        mlp_simple.fit([train_DWI], [train_prevDirection, train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ###
    elif (modelToUse == 'cnn_special_pd'):
        f = h5py.File(pTrainData, "r")
        train_DWI_past = np.array(f["train_DWI_prev"].value)
        f.close()
        
        print(str(train_DWI.shape))
        train_DWI = np.reshape(train_DWI, [noSamples,16,16])
        print(str(train_DWI.shape))

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
       
    elif (modelToUse == 'mlp_doubleIn_single'):
        ### mirror input/output to enable tracking in two directions ###

        ### fire
        mlp_simple = nn_helper.get_mlp_multiInput_singleOutput(loss=loss, lr=lr, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], inputShapeVector=(3,), outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs, normalizeOutput = unitTangent)       

        train_DWI_repeated = np.concatenate((train_DWI,train_DWI) )
        tensionInput = np.concatenate( (-1*train_prevDirection, train_nextDirection) )
        tensionOutput = np.concatenate( (train_nextDirection, -1*train_prevDirection) )
        mlp_simple.fit([train_DWI_repeated, tensionInput], [tensionOutput], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
        
        ### ### ###
    ### MLP SINGLE ###
        ### ### ###
    elif (modelToUse == 'mlp_single'):
        class_weight = None
        
        if(loss == 'sqCos2WEP'):
            noSamples = len(train_DWI)
            labels = np.zeros((noSamples,1))
            labels[idx1] = 1
            loss = 'sqCos2'
            
            noPosSamples = len(np.where(labels == 1)[0])
            noNegSamples = len(np.where(labels == 0)[0])
            class_weight = {1: (noPosSamples+noNegSamples) / noPosSamples,
                            0: (noPosSamples+noNegSamples) / noNegSamples}
            class_weight = { 'signLayer': {1: (noPosSamples+noNegSamples) / noPosSamples, 0: (noPosSamples+noNegSamples) / noNegSamples} }
            print(class_weight)
            print(str(train_nextDirection.shape))
            print(str(labels.shape))
        
            mlp_simple = nn_helper.get_mlp_singleOutputWEP(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs, normalizeOutput = unitTangent)  
            mlp_simple.summary()
            mlp_simple.fit([train_DWI], [train_nextDirection, labels], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger], class_weight=class_weight)
        else:
            mlp_simple = nn_helper.get_mlp_singleOutput(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs, normalizeOutput = unitTangent)  
            mlp_simple.summary()
            mlp_simple.fit([train_DWI], [train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
            
    ###
        ### ### ###
    ### 2MLP SINGLE ###
        ### ### ###    
    elif (modelToUse == '2mlp_single'):
        # load aggregated previous dwi coeffs
        f = h5py.File(pTrainData, "r")
        train_DWI_pastAgg = np.array(f["train_DWI_pastAgg"].value)
        f.close()

        if(loss == 'sqCos2WEP'):
            noSamples = len(train_DWI)
            labels = np.zeros((noSamples,1))
            labels[idx1] = 1
            loss = 'sqCos2'
        
        mlp_simple = nn_helper.get_2mlp_singleOutputWEP(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs, normalizeOutput = unitTangent)  
        
        noPosSamples = len(np.where(labels == 1)[0])
        noNegSamples = len(np.where(labels == 0)[0])
        class_weight = {1: (noPosSamples+noNegSamples) / noPosSamples,
                        0: 10* (noPosSamples+noNegSamples) / noNegSamples}
        print(class_weight)

        class_weight = { 'signLayer': {1: (noPosSamples+noNegSamples) / noPosSamples, 0: (noPosSamples+noNegSamples) / noNegSamples} }
        mlp_simple.summary()
        mlp_simple.fit([train_DWI, train_DWI_pastAgg], [train_nextDirection, labels], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])#, class_weight=class_weight)
    ###
    
    elif (modelToUse == 'unet'):
        cnn_simple = nn_helper.get_3Dunet_simpleTracker(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=train_DWI.shape[1:5], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs)       
        cnn_simple.fit([train_DWI], [train_prevDirection, train_nextDirection], batch_size=batch_size, epochs=epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])

        
if __name__ == "__main__":
    main()

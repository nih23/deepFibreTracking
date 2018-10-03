from keras.layers import Dense, Activation, Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda, Flatten, BatchNormalization, UpSampling1D, LeakyReLU, PReLU, Dropout, AveragePooling1D, Reshape, Permute, Add, ELU, Conv3D, MaxPooling3D, UpSampling3D, Conv2D
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.models import Model, load_model
from keras.constraints import nonneg
from keras import optimizers, losses
from keras import backend as K
from keras.utils import multi_gpu_model
import keras

from src.SelectiveDropout import SelectiveDropout
import sys, getopt
import tensorflow as tf
import h5py
import numpy as np
import time


from keras import backend as K


def swish(x, c = 0.1):
    '''
    "soft" relu function
    see https://openreview.net/pdf?id=Hkuq2EkPf (ICLR2018)
    '''
    return x * K.sigmoid(tf.constant(c, dtype=tf.float32) * x)


def relu_advanced(x):
    '''
    cropped relu function
    '''
    return K.relu(x, max_value=1)


def get_mlp_simpleTracker(inputShapeDWI, outputShape = 3, depth=1, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, useBN=False, useDropout=False, pDropout=0.5):
    '''
    predict direction of past/next streamline position using simple MLP architecture
    Input: DWI subvolume centered at current streamline position
    '''
    inputs = Input(inputShapeDWI)
    layers = [inputs]
    layers.append(Flatten()(layers[-1]))
    
    for i in range(1,depth+1):
        layers.append(Dense(features)(layers[-1]))
        
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        
        layers.append(activation_function(layers[-1]))
        
        if(useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
    
    i1 = layers[-1]
    
    layers.append(Dense(outputShape, kernel_initializer = 'he_normal', name='prevDirection')(i1))
    #layers.append( Lambda(lambda x: x / K.sqrt(K.sum(x ** 2)))(layers[-1]) ) # normalize output to unit vector
    layerPrevDirection = layers[-1]
    
    layers.append(Dense(outputShape, kernel_initializer = 'he_normal', name='nextDirection')(i1))
    #layers.append( Lambda(lambda x: x / K.sqrt(K.sum(x ** 2)))(layers[-1]) ) # normalize output to unit vector
    layerNextDirection = layers[-1]
        
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)
    mlp = Model((layers[0]), outputs=(layerPrevDirection,layerNextDirection))
    mlp.compile(loss=[losses.mse,losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    #mlp.compile(loss=[losses.cosine_proximity,losses.cosine_proximity], optimizer=optimizer) # use in case of directional vectors
    
    return mlp


def get_3Dunet_simpleTracker(inputShapeDWI,outputShape = 3, kernelSz = 3, depth=5, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, pDropout=0.5, poolSz=(2,2,2), useDropout = False, useBN = False):
    '''
    predict direction of past/next streamline position using UNet architecture
    Input: DWI subvolume centered at current streamline position
    '''
    
    inputs = Input(inputShapeDWI)
    
    layersEncoding = []
    layers = [inputs]


    # DOWNSAMPLING STREAM
    for i in range(1,depth+1):
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        if(useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        if(useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        
        
        layersEncoding.append(layers[-1])
        layers.append(MaxPooling3D(pool_size=poolSz)(layers[-1]))

    # ENCODING LAYER
    layers.append(Conv3D(features, kernelSz, padding='same')(layers[-1]))
    if(useBN):
        layers.append(BatchNormalization()(layers[-1]))
    if(useDropout):
        layers.append(Dropout(0.5)(layers[-1]))
    layers.append(activation_function(layers[-1]))
    
    layers.append(Conv3D(features, kernelSz, padding='same')(layers[-1]))
    if(useBN):
        layers.append(BatchNormalization()(layers[-1]))    
    if(useDropout):
        layers.append(Dropout(0.5)(layers[-1]))
    layers.append(activation_function(layers[-1]))

    # UPSAMPLING STREAM
    for i in range(1,depth+1):
        layers.append(concatenate([UpSampling3D(size=poolSz)(layers[-1]), layersEncoding[-i]]))

        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(activation_function(layers[-1]))

        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(activation_function(layers[-1]))

    # final prediction layer
    if(subsampleData):
        layers.append(UpSampling3D(size=8)(layers[-1]))
        
    classificationLayer = layers[-1]
        
    layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(classificationLayer))
    layers.append(activation_function(layers[-1]))
    layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Flatten()(layers[-1]))
    
    # streamline direction prediction
    layers.append(Dense(outputShape,name='prevDirection')(layers[-1]))
    o1 = layers[-1]
    layers.append(Dense(outputShape,name='nextDirection')(layers[-2]))
    o2 = layers[-1] 
    
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)
    u_net_serial = Model(inputs=(layers[0]), outputs=(o1,o2))
    u_net_serial.compile(loss=[losses.mse,losses.mse], optimizer=optimizer)
    
    return u_net_serial
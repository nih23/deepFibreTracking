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



def normalizeDWI(data):
    data = data - np.min(data)
    data /= np.max(data)
    return data


def normalizeStreamlineOrientation(vecs):
    vecNorms = np.sqrt(np.sum(vecs ** 2 , axis = 1))
    vecs = np.nan_to_num(vecs / vecNorms[:,None])   
    vecs = (vecs + 1) / 2
    return vecs


def denormalizeStreamlineOrientation(vecs):
    return 2 * vecs - 1


def get_msd_simplified_trackerNetwork(inputShapeDWI,depth=5, features=64, activation_function=PReLU(), lr=1e-4, noGPUs=3, decayrate=0, pDropout=0.25, subsampleData=False, initialDilationOffset = 0):
    '''
    predict direction to next streamline point
    Input: DWI block
    '''
    layersEncoding = []
    inputs = Input(inputShapeDWI)
    layers = [inputs]
    if(subsampleData):
        layers.append(AveragePooling3D(pool_size=8)(layers[-1]))
    inLayer = layers[-1]
    
    # Mixed-scale Dense ConvNet architecture
    # Pelt, D. M., & Sethian, J. A. (2018). A mixed-scale dense convolutional neural network for image analysis. Proceedings of the National Academy of Sciences, 115(2), 254–259. https://doi.org/10.1073/pnas.1715832114
    for i in range(1,depth+1):
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal', dilation_rate = i+initialDilationOffset)(inLayer))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal', dilation_rate = i+initialDilationOffset)(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layersEncoding.append(layers[-1])

    layers.append(concatenate(layersEncoding))

    if(subsampleData):
        layers.append(UpSampling3D(size=8)(layers[-1]))

        
    # PREDICT NEXT DIRECTION
    layers.append(Conv3D(features, kernelSz, padding='same')(layers[-1])) 
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(BatchNormalization()(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Conv3D(features, kernelSz, padding='same')(layers[-1]))
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(BatchNormalization()(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(3,activation='linear',name='nextStreamlineDirection')(layers[-1]))
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)
    msd_serial = Model(inputs=(layers[0]), outputs=(layers[-1]))
    msd_gpu = multi_gpu_model(msd_serial, gpus=noGPUs)
    msd_gpu.compile(loss=[losses.cosine_proximity], optimizer=optimizer)

    return msd_gpu


def get_msd_advancedTracker(inputShape1,inputShape2,noCrossings = 3, depth=5, features=64, activation_function=PReLU(), lr=1e-4, noGPUs=3, decayrate=0, pDropout=0.25, subsampleData=False, initialDilationOffset = 0):
    '''
    predict direction of next streamline position and likely directions to all adjacent streamline positions
    Input: DWI block, last directional vector
    '''
    layersEncoding = []
    inputs = Input(inputShape1)
    layers = [inputs]
    if(subsampleData):
        layers.append(AveragePooling3D(pool_size=8)(layers[-1]))
    inLayer = layers[-1]
    
    # Mixed-scale Dense ConvNet architecture
    # Pelt, D. M., & Sethian, J. A. (2018). A mixed-scale dense convolutional neural network for image analysis. Proceedings of the National Academy of Sciences, 115(2), 254–259. https://doi.org/10.1073/pnas.1715832114
    for i in range(1,depth+1):
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal', dilation_rate = i+initialDilationOffset)(inLayer))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal', dilation_rate = i+initialDilationOffset)(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layersEncoding.append(layers[-1])

    layers.append(concatenate(layersEncoding))

    if(subsampleData):
        layers.append(UpSampling3D(size=8)(layers[-1]))

        
    # PREDICT LIKELY DIRECTIONS
    layers.append(Conv3D(features, kernelSz, padding='same')(layers[-1])) # at this layer we want to have some kind of prediction of likely directions
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(BatchNormalization()(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Conv3D(features, kernelSz, padding='same')(layers[-1]))
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(BatchNormalization()(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(noCrossings * 2 * 3,activation='linear')(layers[-1]))
    layers.append(Reshape((2*noCrossings, 3), name = 'sl_all_dir')(layers[-1]))
    o_likelyDirections = layers[-1]
    
    # PREDICT NEXT DIRECTION
    layers.append(Conv1D(3,1,activation='sigmoid', padding='same')(layers[-1]))
    auxiliary_input = Input(shape=(inputShape2), name='aux_input')
    x = concatenate([layers[-1], auxiliary_input], axis = 1)
    layers.append(x)
    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(3,activation='tanh',name='sl_next_dir')(layers[-1]))
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)
    msd_serial = Model(inputs=(layers[0],auxiliary_input), outputs=(o_likelyDirections, layers[-1]))
    msd_gpu = multi_gpu_model(msd_serial, gpus=noGPUs)
    msd_gpu.compile(loss=[losses.mse,losses.mse], optimizer=optimizer)

    return msd_gpu


def get_msd_plainTracker(inputShape1,inputShape2,noCrossings = 3, depth=5, features=64, activation_function=PReLU(), lr=1e-4, noGPUs=3, decayrate=0, pDropout=0.25, subsampleData=False, initialDilationOffset = 0):
    '''
    predict likely directions to adjacent streamline positions
    Input: DWI block
    '''
    layersEncoding = []
    inputs = Input(inputShape1)
    layers = [inputs]
    if(subsampleData):
        layers.append(AveragePooling3D(pool_size=8)(layers[-1]))
    inLayer = layers[-1]
    
    # Mixed-scale Dense ConvNet architecture
    # Pelt, D. M., & Sethian, J. A. (2018). A mixed-scale dense convolutional neural network for image analysis. Proceedings of the National Academy of Sciences, 115(2), 254–259. https://doi.org/10.1073/pnas.1715832114
    for i in range(1,depth+1):
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal', dilation_rate = i+initialDilationOffset)(inLayer))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal', dilation_rate = i+initialDilationOffset)(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layersEncoding.append(layers[-1])

    layers.append(concatenate(layersEncoding))

    if(subsampleData):
        layers.append(UpSampling3D(size=8)(layers[-1]))

        
    layers.append(Conv3D(features, kernelSz, padding='same')(layers[-1])) # at this layer we want to have some kind of prediction of likely directions
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(BatchNormalization()(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Conv3D(features, kernelSz, padding='same')(layers[-1]))
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(BatchNormalization()(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(noCrossings * 2 * 3,activation='linear')(layers[-1]))
    layers.append(Reshape((2*noCrossings, 3), name = 'sl_all_dir')(layers[-1]))
    o_likelyDirections = layers[-1]
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)
    msd_serial = Model(inputs=(layers[0]), outputs=(o_likelyDirections))
    msd_gpu = multi_gpu_model(msd_serial, gpus=noGPUs)
    msd_gpu.compile(loss=[losses.mse], optimizer=optimizer)

    return msd_gpu


def get_mlp_simpleTracker(inputShapeDWI,depth=1,features=64,activation_function=LeakyReLU(alpha=0.3),lr=1e-4,noGPUs=4,decayrate=0,pDropout=0.5,avPoolSz=8):
    inputs = Input(inputShapeDWI)
    layers = [inputs]
    layers.append(Flatten()(layers[-1]))
    
    for i in range(1,depth+1):
        layers.append(Dense(features)(layers[-1]))
        #layers.append(BatchNormalization()(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Dropout(0.5)(layers[-1]))
        #layers.append(keras.layers.core.Activation('relu')(max_value = 1)(layers[-1]))
    
    layers.append(Dense(3,activation='linear',name='finalPrediction')(layers[-1]))

    optimizer = optimizers.Adam(lr=lr, decay=decayrate)
    u_net = Model(layers[0], outputs=layers[-1])
    
    u_net.compile(loss=[losses.cosine_proximity], optimizer=optimizer)

    return u_net



def get_3Dunet_simpleTracker(inputShapeDWI,depth=5,features=64,activation_function=LeakyReLU(alpha=0.3),lr=1e-4,noGPUs=4,decayrate=0,pDropout=0.25,subsampleData=False,avPoolSz=8,poolSz=(2,2,2)):
    '''
    predict direction of next streamline position
    Input: DWI block
    '''
    
    inputs = Input(inputShapeDWI)
    
    layersEncoding = []
    layers = [inputs]
    
    if(subsampleData):
        layers.append(AveragePooling3D(pool_size=avPoolSz)(layers[-1]))

    # DOWNSAMPLING STREAM
    for i in range(1,depth+1):
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        #layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(Dropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        #layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(Dropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layersEncoding.append(layers[-1])
        layers.append(MaxPooling3D(pool_size=poolSz)(layers[-1]))

    # ENCODING LAYER
    layers.append(Conv3D(features, kernelSz, padding='same')(layers[-1]))
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Conv3D(features, kernelSz, padding='same')(layers[-1]))
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(activation_function(layers[-1]))

    # UPSAMPLING STREAM
    for i in range(1,depth+1):
        #j = depth - i + 1
        layers.append(concatenate([UpSampling3D(size=poolSz)(layers[-1]), layersEncoding[-i]]))
        #layers.append(UpSampling3D(size=poolSz)(layers[-1]))
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        #ayers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        #ayers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))

    # final prediction layer
    if(subsampleData):
        layers.append(UpSampling3D(size=8)(layers[-1]))
    #layers.append(Conv1D(3,1,activation='sigmoid', padding='same')(layers[-1]))
    layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
    #ayers.append(BatchNormalization()(layers[-1]))
    #layers.append(SelectiveDropout(0.5)(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
    #ayers.append(BatchNormalization()(layers[-1]))
    #layers.append(SelectiveDropout(0.5)(layers[-1]))
    layers.append(activation_function(layers[-1]))
    
    # align with directional vector
    layers.append(Flatten()(layers[-1]))
    #layers.append(Dense(512,activation='linear')(layers[-1]))
    layers.append(Dense(3,activation='linear',name='nextStreamlineDirection')(layers[-1]))
    o2 = layers[-1]
    
    
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)
    u_net_serial = Model(inputs=(layers[0]), outputs=o2)
    unet_multi_gpu = multi_gpu_model(u_net_serial, gpus=noGPUs)
    unet_multi_gpu.compile(loss=[losses.cosine_proximity], optimizer=optimizer)
    
    return unet_multi_gpu


def get_3Dunet_advancedTracker(inputShapeDWI,inputShapeStreamline,depth=5,features=64,activation_function=LeakyReLU(alpha=0.3),lr=1e-4,noGPUs=4,decayrate=0,pDropout=0.25,subsampleData=False,avPoolSz=8,poolSz=(2,2,2)):
    '''
    predict direction of next streamline position and likely directions to all adjacent streamline positions
    Input: DWI block, last directional vector
    '''
    
    inputs = Input(inputShapeDWI)
    
    layersEncoding = []
    layers = [inputs]
    
    if(subsampleData):
        layers.append(AveragePooling3D(pool_size=avPoolSz)(layers[-1]))

    # DOWNSAMPLING STREAM
    for i in range(1,depth+1):
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layersEncoding.append(layers[-1])
        layers.append(MaxPooling3D(pool_size=poolSz)(layers[-1]))

    # ENCODING LAYER
    layers.append(Conv3D(features, kernelSz, padding='same')(layers[-1]))
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Conv3D(features, kernelSz, padding='same')(layers[-1]))
    #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
    layers.append(activation_function(layers[-1]))

    # UPSAMPLING STREAM
    for i in range(1,depth+1):
        #j = depth - i + 1
        layers.append(concatenate([UpSampling3D(size=poolSz)(layers[-1]), layersEncoding[-i]]))
        #layers.append(UpSampling3D(size=poolSz)(layers[-1]))
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))

    # final prediction layer
    if(subsampleData):
        layers.append(UpSampling3D(size=8)(layers[-1]))
    #layers.append(Conv1D(3,1,activation='sigmoid', padding='same')(layers[-1]))
    layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(BatchNormalization()(layers[-1]))
    #layers.append(SelectiveDropout(0.5)(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(BatchNormalization()(layers[-1]))
    #layers.append(SelectiveDropout(0.5)(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Conv3D(3,(1,1,1),activation='linear', padding='same')(layers[-1]))
    
    # align with directional vector
    layers.append(Flatten()(layers[-1]))
    layers.append(Reshape((np.prod(inputShapeDWI[0:3]), 3))(layers[-1])) 
    
    # integrate last streamline direction
    auxiliary_input = Input(shape=(inputShapeStreamline), name='aux_input')
    x = concatenate([layers[-1], auxiliary_input], axis = 1)

    ##layers.append(Conv1D(features,kernelSz, padding='same')(layers[-1]))
    ##layers.append(activation_function(layers[-1]))
    ##layers.append(Conv1D(features,kernelSz, padding='same')(layers[-1]))
    ##layers.append(activation_function(layers[-1]))                  

    # magic
    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(512,activation='relu')(layers[-1]))
    layers.append(Dense(3,activation='relu',name='nextStreamlineDirection')(layers[-1]))
    o2 = layers[-1]
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)
    u_net_serial = Model(inputs=(layers[0],auxiliary_input), outputs=o2)
    #u_net.compile(loss=[losses.mse], optimizer=optimizer)
    
    #msd_serial = Model(inputs=(layers[0]), outputs=(o_likelyDirections))
    unet_multi_gpu = multi_gpu_model(u_net_serial, gpus=noGPUs)
    unet_multi_gpu.compile(loss=[losses.cosine_proximity], optimizer=optimizer)
    
    return unet_multi_gpu

def get_3Dunet_withFixedInnerShape(inputShape,depth=5,features=64,activation_function=LeakyReLU(alpha=0.3),lr=1e-4,noGPUs=4,decayrate=0,pDropout=0.25,subsampleData=False,avPoolSz=8,poolSz=(2,2,2),innerShape=(6,3)):
    inputs = Input(inputShape)
    
    outputs = []
    
    layersEncoding = []
    layers = [inputs]
    
    if(subsampleData):
        layers.append(AveragePooling3D(pool_size=avPoolSz)(layers[-1]))

    # DOWNSAMPLING STREAM
    for i in range(1,depth+1):
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5,dropoutEnabled=1)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layersEncoding.append(layers[-1])
        layers.append(MaxPooling3D(pool_size=poolSz)(layers[-1]))

    # ENCODING LAYER
    noInnerFeatures = np.prod(innerShape)   
    outputShape = np.insert(np.divide(inputShape[0:3], np.power(poolSz,depth)), 3,features).astype(np.int32)
    #äoutputShape = np.divide(inputShape[0:3], np.power(poolSz,depth)).astype(np.int32)
    noOutputFeatures = np.abs(np.prod(outputShape).astype(np.int32))
    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(noInnerFeatures)(layers[-1]))
    layers.append(activation_function(layers[-1]))
    outputs.append(layers[-1])
    #layers.append(Dense(noInnerFeatures)(layers[-1]))
    #layers.append(activation_function(layers[-1]))
    layers.append(Dense(noOutputFeatures)(layers[-1]))
    layers.append(Reshape(outputShape)(layers[-1]))
    
    # UPSAMPLING STREAM
    for i in range(1,depth+1):
        #j = depth - i + 1
        layers.append(concatenate([UpSampling3D(size=poolSz)(layers[-1]), layersEncoding[-i]]))
        #layers.append(UpSampling3D(size=poolSz)(layers[-1]))
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        layers.append(Conv3D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(BatchNormalization()(layers[-1]))
        #layers.append(SelectiveDropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))

    if(subsampleData):
        layers.append(UpSampling3D(size=8)(layers[-1]))
    ##layers.append(Conv1D(features,kernelSz, padding='same')(layers[-1]))
    ##layers.append(activation_function(layers[-1]))
    ##layers.append(Conv1D(features,kernelSz, padding='same')(layers[-1]))
    ##layers.append(activation_function(layers[-1]))                  
    layers.append(Conv3D(288,(1,1,1),activation='linear', padding='same', name='dwiRecon')(layers[-1]))
    #layers.append(Flatten()(layers[-1]))
    #layers.append(Dense(3,activation='linear',name='finalPrediction')(layers[-1]))
    outputs.append(layers[-1])
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)
    u_net = Model(layers[0], outputs=outputs)
    u_net.compile(loss=[losses.mean_absolute_error,losses.mean_absolute_error], optimizer=optimizer)
    return u_net

def activateAllDropoutLayers(m):
    ll = [item for item in m.layers if type(item) is SelectiveDropout]
    for ditLayer in ll:
        ditLayer.setDropoutEnabled(true)


def predict_with_uncertainty(model, x, n_iter=10):
    activateAllDropoutLayers(model)
    result = np.zeros((n_iter,) + x.shape)

    for iter in range(n_iter):
        result[iter] = model.predict(x)

    prediction = result.mean(axis=0)
    uncertainty = result.var(axis=0)
    return prediction, uncertainty, result
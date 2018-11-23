from keras.layers import Dense, Activation, Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda, Flatten, BatchNormalization, UpSampling2D, LeakyReLU, PReLU, Dropout, AveragePooling1D, Reshape, Permute, Add, ELU, Conv3D, MaxPooling3D, UpSampling3D, Conv2D, MaxPooling2D
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


def setAllDropoutLayers(m, value):
    ll = [item for item in m.layers if type(item) is SelectiveDropout]
    for ditLayer in ll:
        ditLayer.setDropoutEnabled(value)
        
        
def printDropoutLayersState(m):
    ll = [item for item in m.layers if type(item) is SelectiveDropout]
    for ditLayer in ll:
        print(ditLayer._getDropoutEnabled())



def squared_cosine_proximity_2(y_true, y_pred):
    '''
    squares cosine loss function (variant 2)
    This loss function allows the network to be invariant wrt. to the streamline orientation. The direction of a vector v_i (forward OR backward (-v_i)) doesn't affect the loss.
    '''
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -  (K.sum(y_true * y_pred, axis=-1) ** 2)


def squared_cosine_proximity_2_withL2penalty(y_true, y_pred):
    '''
    squares cosine loss function (variant 2)
    This loss function allows the network to be invariant wrt. to the streamline orientation. The direction of a vector v_i (forward OR backward (-v_i)) doesn't affect the loss.
    '''
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -(K.sum(y_true * y_pred, axis=-1) ** 2) + (K.sum(y_pred**2, axis=-1) ** 2)


#### learnable activation layer
from keras.engine.base_layer import Layer
class LearnableSwishActivation(Layer):
    '''
    evaluate swish function using
        import keras.backend as K
        import tensorflow as tf
        import src.nn_helper as nn_helper
        importlib.reload(nn_helper)
        from src.nn_helper import swish as swish
        x = tf.linspace(-5.,100.,100).eval(session=K.get_session())
        y = swish(x,c=0.1,b=-10).eval(session=K.get_session())
        y[0:10]
    '''
    def __init__(self, **kwargs):
        super(LearnableSwishActivation, self).__init__(**kwargs)
        self.__name__ = 'learnableSWISH'
        
    def build(self, input_shape):
        self.output_dim = input_shape[1] 
        self.W = self.add_weight(shape=(1,), # Create a trainable weight variable for this layer.
                                 initializer='one', trainable=True, name="swish_c")
        super(LearnableSwishActivation, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x, mask=None):
        return x * K.sigmoid(self.W * x)
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)



def swish(x, c = 0.1, b = 0):
    '''
    "soft" relu function
    see https://openreview.net/pdf?id=Hkuq2EkPf (ICLR2018)
    '''
    return (x) * K.sigmoid(tf.constant(c, dtype=tf.float32) * (x))


def cropped_relu(x):
    '''
    cropped relu function
    '''
    return K.relu(x, max_value=1)


def get_mlp_multiInput_detectEndingStreamlines(inputShapeDWI, inputShapeVector, loss='mse', depth=1, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, useBN=False, useDropout=False, pDropout=0.5):
    '''
    predict direction of past/next streamline position using simple MLP architecture
    Input: DWI subvolume centered at current streamline position
    '''
    i1 = Input(inputShapeDWI)
    layers = [i1]
    layers.append(Flatten()(layers[-1]))
    
    i2 = Input(inputShapeVector)
    
    layers.append(concatenate(  [layers[-1], i2], axis = -1))
    
    for i in range(1,depth+1):
        layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
        
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        
        layers.append(activation_function(layers[-1]))
        
        if(useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
    
    i1 = layers[-1]
    
    layers.append(Dense(1, kernel_initializer = 'he_normal', activation='sigmoid')(layers[-1]))
    
    layerPredictStopping = layers[-1]
        
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    mlp = Model([layers[0],i2], outputs=[layerPredictStopping])
    
    if(loss == 'mse'):
        mlp.compile(loss=[losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    
    return mlp

# the cnn multi input architecture leads to some ambiguities.. 
def get_cnn_multiInput_singleOutput(inputShapeDWI, inputShapeVector, loss='mse', outputShape = 3, depth=1, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, useBN=False, useDropout=False, pDropout=0.5, kernelSz=3, poolSz = 2):
    '''
    predict direction of past/next streamline position using simple CNN architecture
    Input: DWI subvolume centered at current streamline position
    '''
    i1 = Input(inputShapeDWI)
    layers = [i1]
    
    i2 = Input(inputShapeVector)
    # apply CNN to patch
    
    layersEncoding = []
    
    # DOWNSAMPLING STREAM
    for i in range(1,depth+1):
        layers.append(Conv2D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        if(useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        
        layers.append(Conv2D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        if(useDropout):
            layers.append(Dropout(0.5)(layers[-1]))

        layersEncoding.append(layers[-1])
        layers.append(MaxPooling2D(pool_size=poolSz)(layers[-1]))

    # ENCODING LAYER
    layers.append(Conv2D(features, kernelSz, padding='same')(layers[-1]))
    if(useBN):
        layers.append(BatchNormalization()(layers[-1]))
    if(useDropout):
        layers.append(Dropout(0.5)(layers[-1]))
    layers.append(activation_function(layers[-1]))
    
    layers.append(Conv2D(features, kernelSz, padding='same')(layers[-1]))
    if(useBN):
        layers.append(BatchNormalization()(layers[-1]))    
    if(useDropout):
        layers.append(Dropout(0.5)(layers[-1]))
    layers.append(activation_function(layers[-1]))
    
    # UPSAMPLING STREAM
    for i in range(1,depth+1):
        layers.append(concatenate([UpSampling2D(size=poolSz)(layers[-1]), layersEncoding[-i]]))
        #layers.append(concatenate([layers[-1], layersEncoding[-i]]))
        layers.append(Conv2D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(activation_function(layers[-1]))

        layers.append(Conv2D(features, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        layers.append(activation_function(layers[-1]))
    
    layers.append(Conv2D(1, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(activation_function(layers[-1]))
    
    # final prediction layer w/ previous input
    layers.append(Flatten()(layers[-1]))
    
    layers.append(concatenate(  [layers[-1], i2], axis = -1))
    
    layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
    if(useBN):
        layers.append(BatchNormalization()(layers[-1]))
    layers.append(activation_function(layers[-1]))

    layers.append(Dense(outputShape, kernel_initializer = 'he_normal')(layers[-1]))
    
    if(outputShape == 3): # euclidean coordinates
        layers.append( Lambda(lambda x: tf.div(x, K.expand_dims( K.sqrt(K.sum(x ** 2, axis = 1)))  ), name='nextDirection')(layers[-1]) ) # normalize output to unit vector 
    layerNextDirection = layers[-1]
        
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    mlp = Model([layers[0],i2], outputs=[layerNextDirection])
    
    if(loss == 'mse'):
        mlp.compile(loss=[losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    elif(loss == 'cos'):
        mlp.compile(loss=[losses.cosine_proximity], optimizer=optimizer) # use in case of directional vectors
    elif(loss == 'sqCos2'):
        mlp.compile(loss=[squared_cosine_proximity_2], optimizer=optimizer)
    
    return mlp


def get_mlp_multiInput_singleOutput(inputShapeDWI, inputShapeVector, loss='mse', outputShape = 3, depth=1, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, useBN=False, useDropout=False, pDropout=0.5, normalizeOutput=True):
    '''
    predict direction of past/next streamline position using simple MLP architecture
    Input: DWI subvolume centered at current streamline position
    '''
    i1 = Input(inputShapeDWI)
    layers = [i1]
    layers.append(Flatten()(layers[-1]))
    
    i2 = Input(inputShapeVector)
    
    layers.append(concatenate(  [layers[-1], i2], axis = -1))
    
    for i in range(1,depth+1):
        layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
        
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        
        layers.append(activation_function(layers[-1]))
        
        if(useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
    
    i1 = layers[-1]
    
    layers.append(Dense(outputShape, kernel_initializer = 'he_normal')(layers[-1]))
    
    if(normalizeOutput): # euclidean coordinates
        layers.append( Lambda(lambda x: tf.div(x, K.expand_dims( K.sqrt(K.sum(x ** 2, axis = 1)))  ), name='nextDirection')(layers[-1]) ) # normalize output to unit vector 
    layerNextDirection = layers[-1]
        
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    mlp = Model([layers[0],i2], outputs=[layerNextDirection])
    
    if(loss == 'mse'):
        mlp.compile(loss=[losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    elif(loss == 'cos'):
        mlp.compile(loss=[losses.cosine_proximity], optimizer=optimizer) # use in case of directional vectors
    elif(loss == 'sqCos2'):
        mlp.compile(loss=[squared_cosine_proximity_2], optimizer=optimizer)
    mlp.summary()
    return mlp

def get_mlp_singleInput_doubleOutput(inputShapeDWI, loss='mse', outputShape = 3, depth=1, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, useBN=False, useDropout=False, pDropout=0.5):
    '''
    predict direction of past/next streamline position using simple MLP architecture
    Input: DWI subvolume centered at current streamline position
    '''
    i1 = Input(inputShapeDWI)
    layers = [i1]
    layers.append(Flatten()(layers[-1]))
    
    #i2 = Input(inputShapeVector)
    
    #layers.append(concatenate(  [layers[-1], i2], axis = -1))
    
    for i in range(1,depth+1):
        layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
        
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        
        layers.append(activation_function(layers[-1]))
        
        if(useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
    
    i1 = layers[-1]
    
    layers.append(Dense(outputShape, kernel_initializer = 'he_normal')(i1))
    #if(outputShape == 3): # euclidean coordinates
    #    layers.append( Lambda(lambda x: tf.div(x, K.expand_dims( K.sqrt(K.sum(x ** 2, axis = 1)))  ), name='nextDirection')(layers[-1]) ) # normalize output to unit vector 
    layerNextDirection = layers[-1]
    
    
    layers.append(Dense(outputShape, kernel_initializer = 'he_normal')(i1))
    
    if(outputShape == 3): # euclidean coordinates
        layers.append( Lambda(lambda x: tf.div(x, K.expand_dims( K.sqrt(K.sum(x ** 2, axis = 1)))  ), name='prevDirection')(layers[-1]) ) # normalize output to unit vector 
    prevNextDirection = layers[-1]
    
    
        
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    #mlp = Model([layers[0],i2], outputs=[prevNextDirection, layerNextDirection])
    mlp = Model([layers[0]], outputs=[prevNextDirection, layerNextDirection])
    
    if(loss == 'mse'):
        mlp.compile(loss=[losses.mse, losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    elif(loss == 'sqCos2'):
        mlp.compile(loss=[squared_cosine_proximity_2, squared_cosine_proximity_2], optimizer=optimizer)
    
    return mlp

def get_mlp_multiInput_singleOutput_v2(inputShapeDWI, inputShapeVector, loss='mse', outputShape = 3, depth=1, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, useBN=False, useDropout=False, pDropout=0.5):
    '''
    predict direction of past/next streamline position using simple MLP architecture
    Input: DWI subvolume centered at current streamline position
    '''
    i1 = Input(inputShapeDWI)
    layers = [i1]
    layers.append(Flatten()(layers[-1]))
    
    previousDirection = Input(inputShapeVector)
    
    layers.append(concatenate(  [layers[-1], previousDirection], axis = -1))
    
    for i in range(1,depth+1):
        layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
        
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        
        layers.append(activation_function(layers[-1]))
        
        if(useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
    
    
    
    layers.append(Dense(outputShape, kernel_initializer = 'he_normal')(layers[-1]))
    
    deviationFromPreviousDirection = layers[-1]
    
    layers.append(Add()([previousDirection,deviationFromPreviousDirection]))
    
    if(outputShape == 3): # euclidean coordinates
        layers.append( Lambda(lambda x: tf.div(x, K.expand_dims( K.sqrt(K.sum(x ** 2, axis = 1)))  ), name='nextDirection')(layers[-1]) ) # normalize output to unit vector 
    layerNextDirection = layers[-1]
        
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    mlp = Model([layers[0],previousDirection], outputs=[layerNextDirection])
    
    if(loss == 'mse'):
        mlp.compile(loss=[losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    elif(loss == 'cos'):
        mlp.compile(loss=[losses.cosine_proximity], optimizer=optimizer) # use in case of directional vectors
    elif(loss == 'sqCos2'):
        mlp.compile(loss=[squared_cosine_proximity_2], optimizer=optimizer)
    elif(loss == 'sqCos2_w_L2penalty'):
        mlp.compile(loss=[squared_cosine_proximity_2_withL2penalty], optimizer=optimizer)
    
    return mlp

def get_mlp_multiInput_singleOutput_v3(inputShapeDWI, inputShapeVector, loss='mse', outputShape = 3, depth=1, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, useBN=False, useDropout=False, pDropout=0.5):
    '''
    predict direction of past/next streamline position using simple MLP architecture
    Input: DWI subvolume centered at current streamline position
    '''
    i1 = Input(inputShapeDWI)
    layers = [i1]
    layers.append(Flatten()(layers[-1]))
    
    previousDirection = Input(inputShapeVector)
    
    layers.append(concatenate(  [layers[-1], previousDirection], axis = -1))
    
    for i in range(1,depth+1):
        layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
        
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        
        layers.append(activation_function(layers[-1]))
        
        if(useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
    
    
    
    layers.append(Dense(outputShape, kernel_initializer = 'he_normal')(layers[-1]))
    
    deviationFromPreviousDirection = layers[-1]
    
    layers.append(Add()([previousDirection,deviationFromPreviousDirection]))
    
    if(outputShape == 3): # euclidean coordinates
        layers.append( Lambda(lambda x: tf.div(x, K.expand_dims( K.sqrt(K.sum(x ** 2, axis = 1)))  ), name='nextDirection')(layers[-1]) ) # normalize output to unit vector 
    layerNextDirection = layers[-1]
        
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    mlp = Model([layers[0],previousDirection], outputs=[layerNextDirection, deviationFromPreviousDirection])
    
    if(loss == 'mse'):
        mlp.compile(loss=[losses.mse,losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    elif(loss == 'cos'):
        mlp.compile(loss=[losses.cosine_proximity,losses.mse], optimizer=optimizer) # use in case of directional vectors
    elif(loss == 'sqCos2'):
        mlp.compile(loss=[squared_cosine_proximity_2,losses.mse], optimizer=optimizer)

    return mlp


def get_mlp_singleOutput(inputShapeDWI, loss='mse', outputShape = 3, depth=1, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, useBN=False, useDropout=False, pDropout=0.5):
    '''
    predict direction of past/next streamline position using simple MLP architecture
    Input: DWI subvolume centered at current streamline position
    '''
    inputs = Input(inputShapeDWI)
    layers = [inputs]
    layers.append(Flatten()(layers[-1]))
    
    for i in range(1,depth+1):
        layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
        
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        
        layers.append(activation_function(layers[-1]))
        
        if(useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
    
    i1 = layers[-1]
    
    layers.append(Dense(outputShape, kernel_initializer = 'he_normal')(layers[-1]))
    
    if(outputShape == 3): # euclidean coordinates
        layers.append( Lambda(lambda x: tf.div(x, K.expand_dims( K.sqrt(K.sum(x ** 2, axis = 1)))  ), name='nextDirection')(layers[-1]) ) # normalize output to unit vector 
    layerNextDirection = layers[-1]
        
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    mlp = Model((layers[0]), outputs=(layerNextDirection))
    
    if(loss == 'mse'):
        mlp.compile(loss=[losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    elif(loss == 'cos'):
        mlp.compile(loss=[losses.cosine_proximity], optimizer=optimizer) # use in case of directional vectors
    elif(loss == 'sqCos'):
        mlp.compile(loss=[squared_cosine_proximity], optimizer=optimizer)
    elif(loss == 'sqCos2'):
        mlp.compile(loss=[squared_cosine_proximity_2], optimizer=optimizer)
    
    return mlp


### APPROXIMATE BAYESIAN DEEP LEARNING MODELS
def get_mlp_multiInput_singleOutput_bayesian(inputShapeDWI, inputShapeVector, loss='mse', outputShape = 3, depth=1, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, useBN=False, pDropout=0.5):
    '''
    predict direction of past/next streamline position using simple MLP architecture
    Input: DWI subvolume centered at current streamline position
    '''
    i1 = Input(inputShapeDWI)
    layers = [i1]
    layers.append(Flatten()(layers[-1]))
    
    i2 = Input(inputShapeVector)
    
    layers.append(concatenate(  [layers[-1], i2], axis = -1))
    
    for i in range(1,depth+1):
        layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
        
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        
        layers.append(activation_function(layers[-1]))
        
        layers.append(SelectiveDropout(0.5, dropoutEnabled = 1)(layers[-1]))
    
    i1 = layers[-1]
    
    layers.append(Dense(outputShape, kernel_initializer = 'he_normal')(layers[-1]))
    
    if(outputShape == 3): # euclidean coordinates
        layers.append( Lambda(lambda x: tf.div(x, K.expand_dims( K.sqrt(K.sum(x ** 2, axis = 1)))  ), name='nextDirection')(layers[-1]) ) # normalize output to unit vector 
    layerNextDirection = layers[-1]
        
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    mlp = Model([layers[0],i2], outputs=[layerNextDirection])
    
    if(loss == 'mse'):
        mlp.compile(loss=[losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    elif(loss == 'cos'):
        mlp.compile(loss=[losses.cosine_proximity], optimizer=optimizer) # use in case of directional vectors
    elif(loss == 'sqCos2'):
        mlp.compile(loss=[squared_cosine_proximity_2], optimizer=optimizer)
    
    return mlp




def get_mlp_singleOutput_bayesian(inputShapeDWI, loss='mse', outputShape = 3, depth=1, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, useBN=False, pDropout=0.5):
    '''
    predict direction of past/next streamline position using simple MLP architecture
    Input: DWI subvolume centered at current streamline position
    '''
    inputs = Input(inputShapeDWI)
    layers = [inputs]
    layers.append(Flatten()(layers[-1]))
    
    for i in range(1,depth+1):
        layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
        
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        
        layers.append(activation_function(layers[-1]))
        
        layers.append(SelectiveDropout(0.5, dropoutEnabled = 1)(layers[-1]))
    
    i1 = layers[-1]
    
    layers.append(Dense(outputShape, kernel_initializer = 'he_normal')(layers[-1]))
    
    if(outputShape == 3): # euclidean coordinates
        layers.append( Lambda(lambda x: tf.div(x, K.expand_dims( K.sqrt(K.sum(x ** 2, axis = 1)))  ), name='nextDirection')(layers[-1]) ) # normalize output to unit vector 
    layerNextDirection = layers[-1]
        
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)

    mlp = Model((layers[0]), outputs=(layerNextDirection))
    
    if(loss == 'mse'):
        mlp.compile(loss=[losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    elif(loss == 'cos'):
        mlp.compile(loss=[losses.cosine_proximity], optimizer=optimizer) # use in case of directional vectors
    elif(loss == 'sqCos'):
        mlp.compile(loss=[squared_cosine_proximity], optimizer=optimizer)
    elif(loss == 'sqCos2'):
        mlp.compile(loss=[squared_cosine_proximity_2], optimizer=optimizer)
    
    return mlp
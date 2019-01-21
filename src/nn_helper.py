from keras.layers import Dense, Activation, Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda, Flatten, BatchNormalization, UpSampling2D, LeakyReLU, PReLU, Dropout, AveragePooling1D, Reshape, Permute, Add, ELU, Conv3D, MaxPooling3D, UpSampling3D, Conv2D, MaxPooling2D, Multiply, LSTM, multiply
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.models import Model, load_model
from keras.constraints import nonneg
from keras import optimizers, losses
from keras import backend as K
from keras.utils import multi_gpu_model
import keras

from src.tied_layers1d import Convolution2D_tied

from src.SelectiveDropout import SelectiveDropout
import sys, getopt
import tensorflow as tf
import h5py
import numpy as np
import time
from keras import backend as K
from keras.layers.merge import add

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

def mse_directionInvariant(y_true, y_pred):
    return K.minimum(K.mean(K.square(y_pred - y_true), axis=-1) , K.mean(K.square(-1 * y_pred - y_true), axis=-1))

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


# the cnn multi input architecture leads to some ambiguities.. 
def get_2DCNN(trainingState, inputShapeDWI, decayrate=0, pDropout=0.5, kernelSz=3, poolSz = (2,2)):
    '''
    predict direction of past/next streamline position using simple CNN architecture
    Input: DWI subvolume centered at current streamline position
    '''
    i1 = Input(inputShapeDWI)
    layers = [i1]
    
    layersEncoding = []
    
    # DOWNSAMPLING STREAM
    for i in range(1,trainingState.depth+1):
        layers.append(Conv2D(trainingState.noFeatures, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        if(trainingState.useBatchNormalization):
            layers.append(BatchNormalization()(layers[-1]))
        layers.append(trainingState.activationFunction(layers[-1]))
        
        layers.append(Conv2D(trainingState.noFeatures, kernelSz, padding='same', kernel_initializer = 'he_normal')(layers[-1]))
        if(trainingState.useBatchNormalization):
            layers.append(BatchNormalization()(layers[-1]))
        layers.append(trainingState.activationFunction(layers[-1]))
            
#        layersEncoding.append(layers[-1])
#        layers.append(MaxPooling2D(pool_size=poolSz)(layers[-1]))


    # final prediction layer w/ previous input
    layers.append(Flatten()(layers[-1]))
    
    layers.append(Dense(trainingState.noFeatures, kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(trainingState.activationFunction(layers[-1]))
    layers.append(Dense(trainingState.noOutputNeurons, kernel_initializer = 'he_normal')(layers[-1]))
    layerNextDirection = layers[-1]
        
    optimizer = optimizers.Adam(lr=trainingState.lr, decay=decayrate)

    mlp = Model([layers[0]], outputs=[layerNextDirection])
    
    if(trainingState.loss == 'mse'):
        mlp.compile(loss=[losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    elif(trainingState.loss == 'cos'):
        mlp.compile(loss=[losses.cosine_proximity], optimizer=optimizer) # use in case of directional vectors
    elif(trainingState.loss == 'sqCos2'):
        mlp.compile(loss=[squared_cosine_proximity_2], optimizer=optimizer)
    
    return mlp

# the cnn multi input architecture leads to some ambiguities.. 
def get_3DCNN(trainingState, inputShapeDWI, decayrate=0, pDropout=0.5, kernelSz=3, poolSz = (2,2,2), dilationRate = (1,1,1)):
    '''
    predict direction of past/next streamline position using simple CNN architecture
    Input: DWI subvolume centered at current streamline position
    '''
    i1 = Input(inputShapeDWI)
    layers = [i1]
    
    layersEncoding = []
    
    # DOWNSAMPLING STREAM
    for i in range(1,trainingState.depth+1):
        layers.append(Conv3D(trainingState.noFeatures, kernelSz, padding='same', kernel_initializer = 'he_normal', dilation_rate = dilationRate)(layers[-1]))
        if(trainingState.useBatchNormalization):
            layers.append(BatchNormalization()(layers[-1]))
        layers.append(trainingState.activationFunction(layers[-1]))
        
        layers.append(Conv3D(trainingState.noFeatures, kernelSz, padding='same', kernel_initializer = 'he_normal', dilation_rate = dilationRate)(layers[-1]))
        if(trainingState.useBatchNormalization):
            layers.append(BatchNormalization()(layers[-1]))
        layers.append(trainingState.activationFunction(layers[-1]))
            
        layersEncoding.append(layers[-1])
#        layers.append(MaxPooling3D(pool_size=poolSz)(layers[-1]))


    # final prediction layer w/ previous input
    layers.append(Flatten()(layers[-1]))
    
    layers.append(Dense(trainingState.noFeatures, kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(trainingState.activationFunction(layers[-1]))
    layers.append(Dense(trainingState.noOutputNeurons, kernel_initializer = 'he_normal')(layers[-1]))
    layerNextDirection = layers[-1]
        
    optimizer = optimizers.Adam(lr=trainingState.lr, decay=decayrate)

    mlp = Model([layers[0]], outputs=[layerNextDirection])
    
    if(trainingState.loss == 'mse'):
        mlp.compile(loss=[losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    elif(trainingState.loss == 'cos'):
        mlp.compile(loss=[losses.cosine_proximity], optimizer=optimizer) # use in case of directional vectors
    elif(trainingState.loss == 'sqCos2'):
        mlp.compile(loss=[squared_cosine_proximity_2], optimizer=optimizer)
    
    return mlp


def get_rcnn(trainingState, inputShapeDWI, decayrate=0, pDropout=0.5, kernelSz=3, poolSz = (2,2)):

    inputs = Input(inputShapeDWI)
    layers = [inputs]

    for i in range(1, trainingState.depth + 1):
        layers.append(RCL_block(layers[-1], activation_function=trainingState.activationFunction, features=trainingState.noFeatures,
                                name="RCL-" + str(i)))
        if(trainingState.useBatchNormalization):
            layers.append(BatchNormalization()(layers[-1]))
        if(trainingState.useDropout):
            layers.append(Dropout(0.5)(layers[-1]))


    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(trainingState.noFeatures, kernel_initializer = 'he_normal')(layers[-1]))
    if(trainingState.useBatchNormalization):
        layers.append(BatchNormalization()(layers[-1]))
    layers.append(trainingState.activationFunction(layers[-1]))
    layers.append(Dense(trainingState.noOutputNeurons, kernel_initializer = 'he_normal')(layers[-1]))

    layerNextDirection = layers[-1]

    optimizer = optimizers.Adam(lr=trainingState.lr, decay=decayrate)
    mlp = Model([layers[0]], outputs=[layerNextDirection])

    if(trainingState.loss == 'mse'):
        mlp.compile(loss=[losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    elif(trainingState.loss == 'cos'):
        mlp.compile(loss=[losses.cosine_proximity], optimizer=optimizer) # use in case of directional vectors
    elif(trainingState.loss == 'sqCos2'):
        mlp.compile(loss=[squared_cosine_proximity_2], optimizer=optimizer)

    return mlp

def RCL_block(l, activation_function=LeakyReLU(), features=32, kernel_size=3, name="RCL"):
    """Build recurrent ConvLayer. See https://doi.org/10.1109/CVPR.2015.7298958 (i.e. Figure 3)
    Parameters
    ----------
    l: Keras Layer (Tensor?)
        Previous layer of the neural network.
    activation_function: Keras Activation Function
        Activation function (standard: LeakyReLU()).
    features: int
        Number of extracted features.
    kernel_size: int
        Size of Convolution Kernel.
    name: string
        Name of the recurrent ConvLayer (standard: 'RCL').
    :param l: Keras Layer (Tensor?)
        Previous layer of the neural network.
    :param activation_function: Keras Activation Function
        Activation function (standard: LeakyReLU()).
    :param features: int
        Number of extracted features.
    :param kernel_size: int
        Size of Convolution Kernel.
    :param name: string
        Name of the recurrent ConvLayer (standard: 'RCL').
    Returns
    -------
    stack15: keras layer stack
        Recurrent ConvLayer as Keras Layer Stack
    :return: stack15: keras layer stack
        Recurrent ConvLayer as Keras Layer Stack
    """
    conv1 = Conv2D(features, kernel_size, padding='same', name=name)
    stack1 = conv1(l)
    stack2 = activation_function(stack1)
    stack3 = BatchNormalization()(stack2)

    # UNROLLED RECURRENT BLOCK(s)
    conv2 = Conv2D(features, kernel_size, padding='same', init='he_normal')
    stack4 = conv2(stack3)
    stack5 = add([stack1, stack4])
    stack6 = activation_function(stack5)
    stack7 = BatchNormalization()(stack6)

    conv3 = Convolution2D_tied(features, kernel_size, padding='same', tied_to=conv2)
    stack8 = conv3(stack7)
    stack9 = add([stack1, stack8])
    stack10 = activation_function(stack9)
    stack11 = BatchNormalization()(stack10)

    conv4 = Convolution2D_tied(features, kernel_size, padding='same', tied_to=conv2)
    stack12 = conv4(stack11)
    stack13 = add([stack1, stack12])
    stack14 = activation_function(stack13)
    stack15 = BatchNormalization()(stack14)

    return stack15



def get_mlp_singleOutput(trainingState, inputShapeDWI, decayrate = 0):
    '''
    predict direction of past/next streamline position using simple MLP architecture
    Input: DWI subvolume centered at current streamline position
    '''
    inputs = Input(inputShapeDWI)
    layers = [inputs]
    layers.append(Flatten()(layers[-1]))
    
    for i in range(1,trainingState.depth+1):
        layers.append(Dense(trainingState.noFeatures, kernel_initializer = 'he_normal')(layers[-1]))
        
        if(trainingState.useBatchNormalization):
            layers.append(BatchNormalization()(layers[-1]))
        
        layers.append(trainingState.activationFunction(layers[-1]))
        
        if(trainingState.useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
    
    i1 = layers[-1]
    layers.append(Dense(trainingState.noFeatures, kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(trainingState.activationFunction(layers[-1]))
    layers.append(Dense(trainingState.noOutputNeurons, kernel_initializer = 'he_normal')(layers[-1]))
    
#    if(trainingState.unitTangent):
#        layers.append( Lambda(lambda x: tf.div(x, K.expand_dims( K.sqrt(K.sum(x ** 2, axis = 1)))  ), name='nextDirection')(layers[-1]) ) # normalize output to unit vector 
    layerNextDirection = layers[-1]
        
    optimizer = optimizers.Adam(lr=trainingState.lr, decay=decayrate)

    mlp = Model((layers[0]), outputs=(layerNextDirection))
    
    if(trainingState.loss == 'mse'):
        mlp.compile(loss=[losses.mse], optimizer=optimizer)  # use in case of spherical coordinates
    elif(trainingState.loss == 'cos'):
        mlp.compile(loss=[losses.cosine_proximity], optimizer=optimizer) # use in case of directional vectors
    elif(trainingState.loss == 'sqCos2'):
        mlp.compile(loss=[squared_cosine_proximity_2], optimizer=optimizer)
    
    return mlp

def get_mlp_singleOutputWEP(trainingState, inputShapeDWI, decayrate=0):
    '''
    predict direction of past/next streamline position using simple MLP architecture
    Input: DWI subvolume centered at current streamline position
    '''
    inputs = Input(inputShapeDWI)
    layers = [inputs]
    
    
    layers.append(Flatten()(layers[-1]))
    #layers.append(Reshape(target_shape=(100,))(layers[-1]))
    #layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
    #layers.append(activation_function(layers[-1]))
    
    #if(useDropout):
    #    layers.append(Dropout(0.5)(layers[-1]))
    
    
    for i in range(0,trainingState.depth):
        layers.append(Dense(trainingState.noFeatures, kernel_initializer = 'he_normal')(layers[-1]))
        
        if(trainingState.useBatchNormalization):
            layers.append(BatchNormalization()(layers[-1]))
        
        layers.append(trainingState.activationFunction(layers[-1]))
        
        if(trainingState.useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
    
    l1 = layers[-1]

    layers.append(Dense(3, kernel_initializer = 'he_normal')(layers[-1]))
    dirLayer = layers[-1]
    
    layers.append(Dense(trainingState.noFeatures, kernel_initializer = 'he_normal')(l1))
    layers.append(trainingState.activationFunction(layers[-1]))
    layers.append(Dense(trainingState.noFeatures, kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(trainingState.activationFunction(layers[-1]))
    layers.append(Dense(1, kernel_initializer = 'he_normal', activation='sigmoid', name = 'signLayer')(layers[-1]))
    signLayer = layers[-1]
    layers.append(concatenate(  [layers[-1], layers[-1], layers[-1]], axis = -1))
    signLayerConcat = layers[-1]
    #o = multiply(dirLayer,signLayer)
    #layers.append(o)
    layers.append(Multiply()([dirLayer,signLayerConcat]))
    layerNextDirection = layers[-1]
    #layers.append(Dense(1, kernel_initializer = 'he_normal', activation='sigmoid', name = 'signLayer')(signLayerConcat))
    #signLayer = layers[-1]
    
        
    optimizer = optimizers.Adam(lr=trainingState.lr, decay=decayrate)

    mlp = Model((layers[0]), outputs=(layerNextDirection, signLayer))
    
    mlp.compile(loss=[losses.mse, losses.binary_crossentropy], optimizer=optimizer)

        
    
    return mlp

def get_2mlp_singleOutputWEP(inputShapeDWI, loss='mse', outputShape = 3, depth=1, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, useBN=False, useDropout=False, pDropout=0.5, normalizeOutput = True):
    '''
    predict direction of past/next streamline position using simple MLP architecture
    Input: DWI subvolume centered at current streamline position
    '''
    
    print('Loss: ' + loss)
    
    inputs = Input(inputShapeDWI)
    inputs2 = Input(inputShapeDWI)
    
    ## DWI coefficients at current streamline position
    layers = [inputs]
    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(activation_function(layers[-1]))
    if(useDropout):
        layers.append(Dropout(0.5)(layers[-1]))
  
    iii1 = layers[-1]
    
    ## aggregated last DWI values
    layers.append(inputs2)
    i3 = layers[-1]
    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(activation_function(layers[-1]))
    if(useDropout):
        layers.append(Dropout(0.5)(layers[-1]))
        
    ## concat and process both jointly
    layers.append(concatenate([layers[-1],iii1]))
    iii2 = layers[-1]
    layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(activation_function(layers[-1]))
    if(useDropout):
        layers.append(Dropout(0.5)(layers[-1]))

    
    for i in range(1,depth+1):
        layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
        
        if(useBN):
            layers.append(BatchNormalization()(layers[-1]))
        
        layers.append(activation_function(layers[-1]))
        
        if(useDropout):
            layers.append(Dropout(0.5)(layers[-1]))
    
    i1 = layers[-1]

    ## this layer proposes a new direction
    layers.append(Dense(3, kernel_initializer = 'he_normal')(i1))
    dirLayer = layers[-1]
    
    ## compute probability that we should continue tracking
    layers.append(Dense(features, kernel_initializer = 'he_normal')(i1))
    layers.append(activation_function(layers[-1]))
    if(useDropout):
        layers.append(Dropout(0.5)(layers[-1]))
    layers.append(Dense(features, kernel_initializer = 'he_normal')(layers[-1]))
    layers.append(activation_function(layers[-1]))
    layers.append(Dense(1, kernel_initializer = 'he_normal', activation='sigmoid', name = 'signLayer')(layers[-1]))    
    signLayer = layers[-1]

    ## multiply probability by predicted direction
    ## this helps us to even output [0,0,0]-tangents right at the last position of a streamline
    layers.append(concatenate(  [signLayer, signLayer, signLayer], axis = -1))
    signLayerConcat = layers[-1]
    layers.append(Multiply()([dirLayer,signLayerConcat]))
    layerNextDirection = layers[-1]
    
    optimizer = optimizers.Adam(lr=lr, decay=decayrate)
    mlp = Model((layers[0], i3), outputs=(layerNextDirection, signLayer))
    mlp.compile(loss=[squared_cosine_proximity_2, binary_crossentropy], optimizer=optimizer)
    
    return mlp


def weighted_binary_crossentropy( y_true, y_pred, weight=0.1) :
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    logloss = -(y_true * K.log(y_pred+K.epsilon()) * weight + (1 - y_true) * K.log(1 - y_pred+K.epsilon()))
    return K.mean( logloss, axis=-1)


def get_mlp_multiInput_singleOutput_v4(inputShapeDWI, inputShapeVector, loss='mse', outputShape = 3, depth=1, features=64, activation_function=LeakyReLU(alpha=0.3), lr=1e-4, noGPUs=4, decayrate=0, useBN=False, useDropout=False, pDropout=0.5, normalizeOutput=True):
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

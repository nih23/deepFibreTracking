"""
This module contains one-dimensional (Recurrent) Convolutional Neural Networks - Classifiers

One-dimensional as in Keras' definition of Conv1D. Kernelsize is always (num_channels, kernel_width)
    -> Convolution only in one direction
    -> channel order doesn't matter
    -> but because of this channel order information is lost

It can be used to classify one-class-problems like epileptic seizure prediction using (raw) EEG data.
Conventional CNN as well as R-CNN are implemented.
Therefore the tied_layers class by Nico Hoffmann (https://github.com/nih23/UKDDeepLearning/tree/master/FunctionalImaging) was used to implement a Recurrent ConvLayer
of the size 4 as in https://doi.org/10.1109/CVPR.2015.7298958

---------------------------------------------------------------------------------------
| The (R-)CNN itself has the following structure:                                     |
|                                                                                     |
| Input ->  (Recurrent) ConvLayers + Pooling  -> Flatten -> Dense -> Dense  -> Output |
|        |-------------------------------------|-----------------------------|        |
|        | C o n v l u t i o n     B l o c k   |  D e n s e      B l o c k   |        |
|        |-------------------------------------|-----------------------------|        |
|        | - Pooling after every               | - Dense1:                   |        |
|        |   2nd ConvLayer                     |     -> activation function: |        |
|        | - Number of feature maps is         |         LeakyReLU           |        |
|        |   reduced after every 2nd ConvLayer |     -> number of Neurons:   |        |
|        | - num_features_new =                |         32                  |        |
|        |        int(num_features_old/2)+1    |- Dense2:                    |        |
|        | - batch normalization is used       |     -> activation function: |        |
|        |   after every ConvLayer             |         Sigmoid             |        |
|        | - LeakyReLU is used as activation   |     -> number of Neurons:   |        |
|        |   function                          |         1                   |        |
|        |-------------------------------------------------------------------|        |
|                                                                                     |
| Compilation parameters:                                                             |
|                                                                                     |
|    - Loss: Binary Crossentropy                                                      |
|    - Optimizer: Adam                                                                |
|    - Learning Rate: 0.001                                                           |
---------------------------------------------------------------------------------------

author: Hildebrand, Raphael <raphael.hildebrand@mailbox.tu-dresden.de>
supervisors: Jens Müller <jens.mueller1@tu-dresden.de>, Nico Hoffmann <nico.hoffmann@tu-dresden.de>,
             Mathias Eberlein <matthias.eberlein@tu-dresden.de>
"""

import os
from keras.layers.merge import add
import keras
from keras.layers import Dense, BatchNormalization
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LeakyReLU
from tied_layers1d import Convolution1D_tied
from keras.layers import Dropout
from keras.models import Model
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from tensorflow import set_random_seed
from numpy.random import seed as random_seed
from sklearn.metrics import roc_curve, auc

class _RecCnn():
    """
    Baseclass:
    This class implements anything that is independent from the topology.
    """
    def __init__(self):
        self.recurrent = True   # whether or not to use recurrent ConvLayers
        self.num_features = 8   # maximum number of extracted feature maps
        self.conv_depth = 8     # number of (recurrent) ConvLayers
        self.seed = 42          # the random seed for reproducible results
        self.save_model = True  # whether or not to save the model and training log automatically
        self.model = None       # Placeholder
        if self.recurrent:
            r_str = 'rcnn'
        else:
            r_str = 'cnn'
        self.path = os.getcwd()+r_str+'_'+str(self.num_features)+'f_'+str(self.conv_depth)+'d_'+str(self.seed)+'s'

    def __repr__(self):
        string = "String-Representation goes here"
        return string

    def fit(self, X, Y, X_val, Y_val, epochs=100):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_eeg_samples)
            The input data.

        y : array-like, shape (n_samples,)
            The target values (class labels).

        :param X: shape (n_samples, n_channels, n_eeg_samples)
            The input data.
        :param Y: shape (n_samples,)
            The target values (class labels)
        :param X_val: shape (n_samples, n_channels, n_eeg_samples)
            The validation input data.
        :param Y_val: shape (n_samples,)
            The validation target values (class labels)
        :param epochs:
            Number of training epochs

        Returns
        -------
        self : returns a trained RecCnn model.

        :return: self : returns a trained RecCnn model.

        """

        # for reproducability
        # note: it still won't be too reproducible if you use GPUs, for more see:
        #       https://github.com/keras-team/keras/issues/2479#issuecomment-213987747
        random_seed(self.seed)
        set_random_seed(self.seed)

        # calculates the class weights of the dataset
        w = len(np.extract(Y == 0, Y))
        w_0 = 1 / (len(np.extract(Y == 0, Y)) / w)
        w_1 = 1 / (len(np.extract(Y == 1, Y)) / w)
        class_weight = {0: w_0, 1: w_1}

        # Save best model only, based on the training loss. Saves model to topology specific file in working directory
        saveBestModel = ModelCheckpoint(self.path + '.h5', monitor='loss', verbose=1, save_best_only=True, mode='auto')

        # Log the training metrics in a topology specific file in the working directory
        csv_logger = CSVLogger(self.path + '_log.csv', append=True, separator=';')

        if self.recurrent:
            self.model = self.build_model_rcnn(X, depth=self.conv_depth, num_features=self.num_features)
        else:
            self.model = self.build_model_cnn(X, depth=self.conv_depth, num_features=self.num_features)

        if self.save_model:
            self.model.fit(x=X, y=Y,
                           batch_size=64,
                           epochs=epochs,
                           verbose=1,
                           class_weight=class_weight,
                           shuffle=True,
                           validation_data=(X_val, Y_val),
                           callbacks=[saveBestModel, csv_logger])
        else:
            self.model.fit(x=X, y=Y,
                           batch_size=64,
                           epochs=epochs,
                           verbose=1,
                           class_weight=class_weight,
                           shuffle=True,
                           validation_data=(X_val, Y_val))

    def predict(self, X, threshold=0.5):
        """Predict using the multi-layer perceptron classifier

        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_eeg_samples)
            The input data.

        thresehold: float
            Threshold value to distinct between class '0' and '1'
            Standard values is threshold = 0.5

        :param X: shape (n_samples, n_channels, n_eeg_samples)
        :param threshold: float

        Returns
        -------
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        :returns: Y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """

        Y = self.model.predict(X)
        for i in range(len(Y)):
            if Y[i] < threshold:
                Y[i] = 0
            else:
                Y[i] = 1

        return Y

    def predict_proba(self, X):
        """Probability estimates.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        :param X: shape (n_samples, n_channels, n_eeg_samples). Input data.

        Returns
        -------
        Y_prob : array-like, shape (n_samples,)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        :returns: Y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes (probability values).
        """

        Y_prob = self.model.predict(X)
        return Y_prob

    def score(self, X, Y):
        """Returns the AUC on the given test data and labels.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        Y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        :param X: shape = (n_samples, n_features). Test samples.
        :param Y: shape = (n_samples) or (n_samples, n_outputs). True labels for X.

        Returns
        -------
        score : float
            AUC of self.predict(X) wrt. y.

        :returns: score : (float) AUC of self.predict(X) wrt. y.
        """
        Y_pred = self.predict(X)
        fpr, tpr, _ = roc_curve(Y, Y_pred)
        roc_auc = auc(fpr, tpr)

        return roc_auc

    def get_params(self):
        """Get parameters for this estimator.
        Parameters
        ----------
        self

        Returns
        -------
        :returns: params : mapping of string to any
            Parameter names mapped to their values.
        """

        return {"recurrent": self.recurrent,
                "num_features": self.num_features,
                "conv_depth": self.conv_depth}

    def set_params(self, recurrent, num_features, conv_depth, save_model):
        """Set the parameters of this estimator.

        Parameters
        ----------
        recurrent: boolean
            Defines if generated CNN uses recurrent conv layers or standard conv layers
        num_features: int
            Maximum number of generated feature maps
        conv_depth: int
            Number of (recurrent) conv layers
        save_model: boolean
            Whether or not to save the model automatically

        :param recurrent: boolean
            Defines if generated CNN uses recurrent conv layers or standard conv layers
        :param num_features: int
            Maximum number of extracted feature maps
        :param conv_depth: int
            Number of (recurrent) ConvLayers
        :param save_model: boolean
            Whether or not to save the model automatically

        Returns
        -------
        self

        :returns: self
        """

        self.recurrent = recurrent
        self.num_features = num_features
        self.conv_depth = conv_depth
        self.save_model = save_model

    def save(self, model_path='model.h5'):
        """Save the model.

        Parameters
        ----------
        model_path: string
            Full or relative path where the model is being saved

        :param model_path: string
            Full or relative path where the model is being saved

        Returns
        -------
        self

        :returns: self
        """
        self.model.save(model_path)

    def load(self, model_path='model.h5'):
        """Load the model.
        Parameters
        ----------
        model_path: string
            Full or relative path of model

        :param model_path: string
            Full or relative path of model

        Returns
        -------
        self

        :returns: self
        """

        self.model = load_model(model_path)

    def build_model_cnn(self, X, verbose=False, en_dropout=False, depth=1, pooling=2, num_features=1):
        """Build non-recurrent ConvNet.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        verbose: boolean
            Verbose level (standard: False)
        en_dropout: boolean
            Whether use dropout regulation or not (standard: True)
        depth: int
            Number of ConvLayers
        pooling: int
            Size of pooling window (standard: 2)
        num_features: int
            Maximum number of extracted feature maps

        :param X: array-like, shape = (n_samples, n_features)
            Test samples.
        :param verbose: boolean
            Verbose level (standard: False)
        :param en_dropout: boolean
            Whether use dropout regulation or not (standard: True)
        :param depth: int
            Number of ConvLayers
        :param pooling: int
            Size of pooling window (standard: 2)
        :param num_features: int
            Maximum number of extracted feature maps

        Returns
        -------
        keras model

        :return: model: keras model
            generated Keras Model
        """

        _, numSamples, numChannels = X.shape
        inputs = Input((numSamples, numChannels))
        layers = [inputs]

        activation_function = LeakyReLU()
        
        kernel_sz = 3
        for i in range(1, depth + 1):
            layers.append(Conv1D(num_features, kernel_sz, padding='same')(layers[-1]))
            layers.append(activation_function(layers[-1]))
            layers.append(BatchNormalization()(layers[-1]))
            if en_dropout:
                layers.append(Dropout(0.5)(layers[-1]))
            if pooling > 0:
                if i % 2 == 0:
                    layers.append(MaxPooling1D(pool_size=(pooling))(layers[-1]))
                    num_features = int(num_features / 2) + 1

        layers.append(Flatten()(layers[-1]))
        layers.append(Dense(32)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        if en_dropout:
            layers.append(Dropout(0.5)(layers[-1]))
        layers.append(Dense(1, activation='sigmoid')(layers[-1]))

        model = Model(layers[0], layers[-1])

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.adam(lr=0.001),
                      metrics=['binary_accuracy'])
        if verbose:
            model.summary()

        return model

    def build_model_rcnn(self, X, verbose=False, en_dropout=False, depth=1, pooling=2, num_features=1):
        """Build recurrent ConvNet.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        verbose: boolean
            Verbose level (standard: False)
        en_dropout: boolean
            Whether use dropout regulation or not (standard: True)
        depth: int
            Number of ConvLayers
        pooling: int
            Size of pooling window (standard: 2)
        num_features: int
            Maximum number of extracted feature maps

        :param X: array-like, shape = (n_samples, n_features)
            Test samples.
        :param verbose: boolean
            Verbose level (standard: False)
        :param en_dropout: boolean
            Whether use dropout regulation or not (standard: True)
        :param depth: int
            Number of ConvLayers
        :param pooling: int
            Size of pooling window (standard: 2)
        :param num_features: int
            Maximum number of extracted feature maps

        Returns
        -------
        keras model

        :return: model: keras model
            generated Keras Model
        """
        _, numSamples, numChannels = X.shape
        inputs = Input((numSamples, numChannels))
        layers = [inputs]

        activation_function = LeakyReLU()

        for i in range(1, depth + 1):
            layers.append(self.RCL_block(layers[-1], activation_function=activation_function, features=num_features,
                                    name="RCL-" + str(i)))
            if en_dropout:
                layers.append(Dropout(0.5)(layers[-1]))
            if pooling > 0:
                if i % 2 == 0:
                    layers.append(MaxPooling1D(pool_size=(pooling))(layers[-1]))
                    num_features = int(num_features / 2) + 1

        layers.append(Flatten()(layers[-1]))
        layers.append(Dense(32)(layers[-1]))
        layers.append(activation_function(layers[-1]))
        if en_dropout:
            layers.append(Dropout(0.5)(layers[-1]))
        layers.append(Dense(1, activation='sigmoid')(layers[-1]))

        model = Model(layers[0], layers[-1])

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.adam(lr=0.001),
                      metrics=['binary_accuracy'])
        if verbose:
            model.summary()
        return model

    def RCL_block(self, l, activation_function=LeakyReLU(), features=32, kernel_size=3, name="RCL"):
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
        conv1 = Conv1D(features, kernel_size, padding='same', name=name)
        stack1 = conv1(l)
        stack2 = activation_function(stack1)
        stack3 = BatchNormalization()(stack2)

        # UNROLLED RECURRENT BLOCK(s)
        conv2 = Conv1D(features, kernel_size, padding='same', init='he_normal')
        stack4 = conv2(stack3)
        stack5 = add([stack1, stack4])
        stack6 = activation_function(stack5)
        stack7 = BatchNormalization()(stack6)

        conv3 = Convolution1D_tied(features, kernel_size, padding='same', tied_to=conv2)
        stack8 = conv3(stack7)
        stack9 = add([stack1, stack8])
        stack10 = activation_function(stack9)
        stack11 = BatchNormalization()(stack10)

        conv4 = Convolution1D_tied(features, kernel_size, padding='same', tied_to=conv2)
        stack12 = conv4(stack11)
        stack13 = add([stack1, stack12])
        stack14 = activation_function(stack13)
        stack15 = BatchNormalization()(stack14)

        return stack15

class RecCnnRCNN_generic(_RecCnn):
    """
    Class for ConvNet or recurrent ConvNet.
    """
    def __init__(self, recurrent, num_features, conv_depth, save_model):
        """Initializes the topology of the (recurrent) ConvNet using the given parameters.

        Parameters
        ----------
        recurrent: boolean
            Defines if generated CNN uses recurrent conv layers or standard conv layers
        num_features: int
            Maximum number of generated feature maps
        conv_depth: int
            Number of (recurrent) conv layers
        save_model: boolean
            Whether or not to save the model automatically

        :param recurrent: boolean
            Defines if generated CNN uses recurrent conv layers or standard conv layers
        :param num_features: int
            Maximum number of extracted feature maps
        :param conv_depth: int
            Number of (recurrent) ConvLayers
        :param save_model: boolean
            Whether or not to save the model automatically

        Returns
        -------
        self

        :returns: self
        """

        _RecCnn.__init__(self)

        self.recurrent = recurrent          # boolean
        self.num_features = num_features    # maximum number of features
        self.conv_depth = conv_depth        # number of convolution layers
        self.save_model = save_model        # save model to working directory
        self.set_params(recurrent=self.recurrent, num_features=self.num_features, conv_depth=self.conv_depth,
                        save_model=self.save_model)


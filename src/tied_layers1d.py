from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.engine import Layer, InputSpec
from keras.utils.conv_utils import conv_output_length
from keras.utils import conv_utils
from keras.layers import Convolution1D, Conv1D
from copy import deepcopy
import tensorflow as tf
from keras.utils.generic_utils import serialize_keras_object
from keras.layers import serialize, deserialize
from keras import layers
import numpy as np

class Convolution1D_tied(Layer):
    '''Convolution operator for filtering neighborhoods of one-dimensional inputs.
    When using this layer as the first layer in a model,
    either provide the keyword argument `input_dim`
    (int, e.g. 128 for sequences of 128-dimensional vectors),
    or `input_shape` (tuple of integers, e.g. (10, 128) for sequences
    of 10 vectors of 128-dimensional vectors).
    # Example
    ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)
        # add a new conv1d on top
        model.add(Convolution1D(32, 3, border_mode='same'))
        # now model.output_shape == (None, 10, 32)
    ```
    # Arguments
        nb_filter: Number of convolution kernels to use
            (dimensionality of the output).
        filter_length: The extension (spatial or temporal) of each filter.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid' or 'same'.
        subsample_length: factor by which to subsample output.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
    # Input shape
        3D tensor with shape: `(samples, steps, input_dim)`.
    # Output shape
        3D tensor with shape: `(samples, new_steps, nb_filter)`.
        `steps` value might have changed due to padding.
    '''
    def __init__(self, filters, kernel_size,strides=1,padding='same',dilation_rate=1,
                 bias_initializer='zeros',kernel_initializer='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample_length=1,
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 use_bias=True, input_dim=None, input_length=None, tied_to=None,data_format='channels_last',rank=1,learnedKernel=None,layer_inner=None,
                 **kwargs):
        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution1D:', border_mode)
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        self.tied_to = tied_to
        self.learnedKernel = np.array(learnedKernel)
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.data_format = data_format
        self.filters = filters
        if self.tied_to is not None:
            self.kernel_size = self.tied_to.kernel_size
        else:
            self.kernel_size = kernel_size
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample_length = subsample_length
        self.subsample = (subsample_length, 1)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_bias = use_bias

        self.rank = 1

        self.layer_inner = layer_inner

        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        #self.layer_inner = kwargs.pop('layer_inner')

        super(Convolution1D_tied, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]

        if self.layer_inner:
            classconfig = self.layer_inner.pop('config')
            classname = self.layer_inner.pop('class_name')
            li_weights = np.asarray(self.layer_inner.pop('weights'))#.pop('value'))
            li_bias = np.asarray(self.layer_inner.pop('bias'))#.pop('value'))


            self.tied_to = layers.deserialize({'class_name': classname,#self.layer_inner.__class__.__name__,
                                'config': classconfig})
            self.tied_to.build(input_shape)
            self.tied_to.set_weights([ li_weights, li_bias ])

        self.regularizers = []
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            new_dim = conv_utils.conv_output_length(
                    input_shape[1],
                    self.kernel_size[0],
                    padding=self.padding,
                    stride=self.strides[0],
                    dilation=self.dilation_rate[0])

            return (input_shape[0],) + tuple([new_dim]) + (self.filters,)

        if self.data_format == 'channels_first':
            return tuple(23)


    def call(self, inputs):
        if self.tied_to is not None:
            outputs = K.conv1d(
               inputs,
               self.tied_to.kernel,
               strides=self.strides[0],
               padding=self.padding,
               data_format=self.data_format,
               dilation_rate=self.dilation_rate[0])
        else:
            # this branch is typically entered when a previously trained model is being loaded again
            outputs = K.conv1d(
               inputs,
               self.learnedKernel,
               strides=self.strides[0],
               padding=self.padding,
               data_format=self.data_format,
               dilation_rate=self.dilation_rate[0])

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding,
                'data_format': self.data_format,
                'dilation_rate': self.dilation_rate,
                'activation': activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'kernel_initializer': initializers.serialize(self.kernel_initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                'kernel_constraint': constraints.serialize(self.kernel_constraint),
                'bias_constraint': constraints.serialize(self.bias_constraint),
                'input_dim': self.input_dim,
                'learnedKernel': self.tied_to.get_weights()[0],
                'input_length': self.input_length}
        config2 = {'layer_inner': {'bias': np.asarray(self.tied_to.get_weights()[1]),
                                   'weights': np.asarray(self.tied_to.get_weights()[0]),
                                   'class_name': self.tied_to.__class__.__name__,
                                   'config': self.tied_to.get_config()}}
        base_config = super(Convolution1D_tied, self).get_config()
        return dict(list(base_config.items()) + list(config.items()) + list(config2.items()))
    
class Convolution2D_tied(Layer):
    '''Convolution operator for filtering neighborhoods of one-dimensional inputs.
    When using this layer as the first layer in a model,
    either provide the keyword argument `input_dim`
    (int, e.g. 128 for sequences of 128-dimensional vectors),
    or `input_shape` (tuple of integers, e.g. (10, 128) for sequences
    of 10 vectors of 128-dimensional vectors).
    # Example
    ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)
        # add a new conv1d on top
        model.add(Convolution1D(32, 3, border_mode='same'))
        # now model.output_shape == (None, 10, 32)
    ```
    # Arguments
        nb_filter: Number of convolution kernels to use
            (dimensionality of the output).
        filter_length: The extension (spatial or temporal) of each filter.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid' or 'same'.
        subsample_length: factor by which to subsample output.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
    # Input shape
        3D tensor with shape: `(samples, steps, input_dim)`.
    # Output shape
        3D tensor with shape: `(samples, new_steps, nb_filter)`.
        `steps` value might have changed due to padding.
    '''
    def __init__(self, filters, kernel_size,strides=(1,1),padding='same',dilation_rate=(1,1),
                 bias_initializer='zeros',kernel_initializer='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample_length=1,
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 use_bias=True, input_dim=None, input_length=None, tied_to=None,data_format='channels_last',rank=2,learnedKernel=None,layer_inner=None,
                 **kwargs):
        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        self.tied_to = tied_to
        self.learnedKernel = np.array(learnedKernel)
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.data_format = data_format
        self.filters = filters
        if self.tied_to is not None:
            self.kernel_size = self.tied_to.kernel_size
        else:
            self.kernel_size = kernel_size
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample_length = subsample_length
        self.subsample = (subsample_length, 1)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_bias = use_bias

        self.rank = 2

        self.layer_inner = layer_inner

        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        #self.layer_inner = kwargs.pop('layer_inner')

        super(Convolution2D_tied, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]

        if self.layer_inner:
            classconfig = self.layer_inner.pop('config')
            classname = self.layer_inner.pop('class_name')
            li_weights = np.asarray(self.layer_inner.pop('weights'))#.pop('value'))
            li_bias = np.asarray(self.layer_inner.pop('bias'))#.pop('value'))


            self.tied_to = layers.deserialize({'class_name': classname,#self.layer_inner.__class__.__name__,
                                'config': classconfig})
            self.tied_to.build(input_shape)
            self.tied_to.set_weights([ li_weights, li_bias ])

        self.regularizers = []
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':

            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = self.filters

            rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                         self.padding, self.strides[0])
            cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                                 self.padding, self.strides[1])
            
            return (input_shape[0], rows, cols, out_filters)
        
        if self.data_format == 'channels_first':
            return tuple(23)


    def call(self, inputs):
        if self.tied_to is not None:
            outputs = K.conv2d(
               inputs,
               self.tied_to.kernel,
               strides=self.strides,
               padding=self.padding,
               data_format=self.data_format,
               dilation_rate=self.dilation_rate)
        else:
            # this branch is typically entered when a previously trained model is being loaded again
            outputs = K.conv2d(
               inputs,
               self.learnedKernel,
               strides=self.strides,
               padding=self.padding,
               data_format=self.data_format,
               dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding,
                'data_format': self.data_format,
                'dilation_rate': self.dilation_rate,
                'activation': activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'kernel_initializer': initializers.serialize(self.kernel_initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                'kernel_constraint': constraints.serialize(self.kernel_constraint),
                'bias_constraint': constraints.serialize(self.bias_constraint),
                'input_dim': self.input_dim,
                'learnedKernel': self.tied_to.get_weights()[0],
                'input_length': self.input_length}
        config2 = {'layer_inner': {'bias': np.asarray(self.tied_to.get_weights()[1]),
                                   'weights': np.asarray(self.tied_to.get_weights()[0]),
                                   'class_name': self.tied_to.__class__.__name__,
                                   'config': self.tied_to.get_config()}}
        base_config = super(Convolution2D_tied, self).get_config()
        return dict(list(base_config.items()) + list(config.items()) + list(config2.items()))

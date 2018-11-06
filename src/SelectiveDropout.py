from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.legacy import interfaces
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

class SelectiveDropout(Layer):

    @interfaces.legacy_dropout_support
    def __init__(self, rate, noise_shape=None, seed=None, dropoutEnabled = 0, **kwargs):
        super(SelectiveDropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True
        self.tf_dropoutEnabled = tf.Variable([dropoutEnabled], name="dropout_enabled")
        tf.global_variables_initializer().run(session=K.get_session())
        K.get_session().run(self.tf_dropoutEnabled)
        
        if(self._getDropoutEnabled() == 0):
            print("[Warning] dropout sampling wasn't activated during initialization")


    def _getDropoutEnabled(self):
        return self.tf_dropoutEnabled.eval(session=K.get_session())


    def setDropoutEnabled(self, enabled):
        print('Warning: changes to the dropout status are effective solely after the first prediction of the network or after saving and re-loading the network!')
        self.tf_dropoutEnabled = tf.assign(self.tf_dropoutEnabled, [int(enabled)])


    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
            def dropped_inputs():
                dropoutRes = K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
                #dropoutRes = tf.Print(dropoutRes, [dropoutRes], 'res = ')
                return dropoutRes
                
            result = tf.cond(tf.squeeze(self.tf_dropoutEnabled) < tf.constant(1), lambda: inputs, lambda: dropped_inputs())
            return result

        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed,
                  'dropoutEnabled': int(self._getDropoutEnabled())
                  }
        base_config = super(SelectiveDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
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
        self.setDropoutEnabled(int(dropoutEnabled))
        #self.dropoutEnabled = self._getDropoutEnabled()
        assignment = tf.assign(self.dropoutEnabled, [int(dropoutEnabled)])
        tf.global_variables_initializer().run(session=K.get_session())
        K.get_session().run(assignment)


    def _getDropoutEnabled(self):
      with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        v = tf.get_variable("dropoutEnabled",shape=[1],dtype=tf.int32)
      return v
#     # workaround for old tensorflow versions that dont know AUTO_REUSE
#      try:
#         with tf.variable_scope("model"):
#            v = tf.get_variable("dropoutEnabled",shape=[1],dtype=tf.int32)
#      except ValueError:
#         with tf.variable_scope("model", reuse=True):
#            v = tf.get_variable("dropoutEnabled",shape=[1],dtype=tf.int32)
#      return v


    def setDropoutEnabled(self, enabled):
        self.dropoutEnabled = self._getDropoutEnabled()
        assignment = tf.assign(self.dropoutEnabled, [int(enabled)])
        K.get_session().run(assignment)


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
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            #return tf.cond(tf.squeeze(self._getDropoutEnabled()) < tf.constant(1), lambda: inputs, lambda: dropped_inputs())
            return tf.cond(tf.squeeze(self._getDropoutEnabled()) < tf.constant(1), lambda: dropped_inputs(), lambda: dropped_inputs())

        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed,
                  'dropoutEnabled': int(self.dropoutEnabled.eval(session=K.get_session()))
                  }
        base_config = super(SelectiveDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
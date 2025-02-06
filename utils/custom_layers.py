import tensorflow as tf
from tensorflow.keras.layers import Layer
##############################################################################
# Custom Layers
##############################################################################

class keras_pad(Layer):
    def call(self, x):
        padding = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        return tf.pad(x, padding, mode='CONSTANT', constant_values=0)

class keras_minimum(Layer):
    def call(self, x, sat_val=1):
        return tf.minimum(x, sat_val)

class keras_floor(Layer):
    def call(self, x):
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)
        return tf.math.floor(x)
    
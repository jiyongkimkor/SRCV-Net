import tensorflow as tf
import tensorflow.keras as keras
from .feature import conv2d, L2

L2 = 1.0e-5
alpha = 0.2


def conv3d_bn(filters, kernel_size, strides, padding, activation):
    conv = keras.layers.Conv3D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding,
                               use_bias=False, kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    leaky_relu = keras.layers.LeakyReLU(alpha=alpha)

    if activation:
        return keras.Sequential([conv, bn, leaky_relu])
    else:
        return keras.Sequential([conv, bn])


class cost_refinement(keras.Model):
    def __init__(self, filters):
        super(cost_refinement, self).__init__()
        self.conv1 = conv3d_bn(filters, 1, 1, 'same', True)
        self.conv2 = conv3d_bn(filters, 3, 1, 'same', True)
        self.conv3 = conv3d_bn(filters, 3, 1, 'same', False)

    def call(self, inputs, training=None, mask=None):
        assert len(inputs) == 2

        x0 = inputs[0]
        
        concat = tf.concat([inputs[0], inputs[1]], -1)

        x1 = self.conv1(concat)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        x4 = x0 + x3
        
        output = x4
        return output  # [N, D, H, W, C]

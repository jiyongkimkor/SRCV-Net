# +
import tensorflow as tf
import tensorflow.keras as keras

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


# +
import tensorflow as tf

class EfficientAttention(tf.keras.layers.Layer):
    def __init__(self, in_channels, key_channels, head_count , value_channels):
        super(EfficientAttention, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        
        self.keys = conv3d_bn(key_channels, 1, 1, 'same', True)
        self.queries = conv3d_bn(key_channels, 1, 1, 'same', True)
        self.values = conv3d_bn(value_channels, 1, 1, 'same', True)
        self.reprojection = conv3d_bn(in_channels, 1, 1, 'same', True)

    def call(self, inputs):
        
        inputs_t = tf.transpose(inputs, (0,4,2,3,1))
        
        n, c, h, w, d = inputs_t.shape
        
        keys = self.keys(inputs_t)
        queries = self.queries(inputs_t)
        values = self.values(inputs_t)

        keys = tf.reshape(keys, (-1, c * h * w, self.key_channels))
        queries = tf.reshape(queries, (-1, c * h * w, self.key_channels))
        values = tf.reshape(values, (-1, c * h * w, self.value_channels))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key_slice = keys[:, :, i * head_key_channels: (i + 1) * head_key_channels]
            query_slice = queries[:, :, i * head_key_channels: (i + 1) * head_key_channels]
            value_slice = values[:, :, i * head_value_channels: (i + 1) * head_value_channels]

            key = tf.nn.softmax(key_slice, axis=1)
            query = tf.nn.softmax(query_slice, axis=2)
            context = tf.matmul(key, value_slice, transpose_a=True)
            attended_value = tf.matmul(context, tf.transpose(query,(0,2,1)))
            attended_value = tf.reshape(attended_value, (-1, c, h, w, head_value_channels))
            attended_values.append(attended_value)

        aggregated_values = tf.concat(attended_values, axis=-1)
        reprojected_value = self.reprojection(aggregated_values)
        
        reprojected_value_t = tf.transpose(reprojected_value, (0,4,2,3,1))
        
        output = inputs + reprojected_value_t
        
        return output
# +
import tensorflow as tf
from tensorflow.keras import layers, models

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = layers.GlobalMaxPooling3D()
        self.avg_pool = layers.GlobalAveragePooling3D()

        self.mlp = models.Sequential([
            layers.Dense(channels // ratio, use_bias=False),
            layers.ReLU(),
            layers.Dense(channels, use_bias=False)
        ])
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        max_pool = self.max_pool(x)
        avg_pool = self.avg_pool(x)

        max_pool = self.mlp(max_pool)
        avg_pool = self.mlp(avg_pool)

        out = max_pool + avg_pool
        out = self.sigmoid(out)
        return out[:, tf.newaxis, tf.newaxis, tf.newaxis] * x


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = (kernel_size - 1) // 2

        self.conv = layers.Conv3D(filters=1, kernel_size=kernel_size, strides=1, padding='same', use_bias=False)
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)

        out = tf.concat([max_pool, avg_pool], axis=-1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out * x


class CBAMBlock(tf.keras.layers.Layer):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def call(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x



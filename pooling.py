import tensorflow as tf

class MaskGlobalMaxPooling1D(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(MaskGlobalMaxPooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x = inputs
        x = x - (1 - mask) * 1e12 # 用一个大的负数mask
        x = tf.reduce_max(x, axis=1, keepdims=True)
        ws = tf.where(inputs == x, x, 0.0)
        ws = tf.reduce_sum(ws, axis=2)
        x = tf.squeeze(x, axis=1)
        return x, ws

class MaskGlobalAveragePooling1D(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(MaskGlobalAveragePooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x = inputs
        x = x * mask
        x = tf.reduce_sum(x, axis=1)
        x =  x / tf.reduce_sum(mask, axis=1)
        ws = tf.square(inputs - tf.expand_dims(x, axis=1))
        ws = tf.reduce_mean(ws, axis=2)
        ws = ws + (1 - mask) * 1e12
        ws = 1 / ws
        return x, ws

class MinVariancePooling(tf.keras.layers.Layer):
    """最小方差加权平均，Inverse-variance weighting
    等价于正太分布的最小熵加权平均"""

    def __init__(self, **kwargs):
        super(MinVariancePooling, self).__init__(**kwargs)

    def build(self, input_shape):
        d = tf.cast(input_shape[2], tf.float32)
        self.alpha = 1 / (d - 1)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        mu = tf.reduce_mean(inputs, axis=2, keepdims=True) # 均值
        var = self.alpha * tf.reduce_sum(tf.square(inputs - mu), axis=2, keepdims=True) # 方差的无偏估计
        var = var + (1 - mask) * 1e12 # 倒数的mask处理
        ivar = 1 / var
        ws = ivar / tf.reduce_sum(ivar, axis=1, keepdims=True)
        return tf.reduce_sum(inputs * ws * mask, axis=1), ws

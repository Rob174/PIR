import tensorflow.keras.backend as K
import tensorflow as tf

class Mish(tf.keras.layers.Layer):
    """
    Mish Activation Function.
    .. math:: https://medium.com/scalian/les-specials-de-yolov4-episode-1-la-backbone-fb818e7e9690#61e9:~:text=La%20fonction%20d%E2%80%99activation%20Mish
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    """

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        custom_config = super(Mish, self).get_config()
        return custom_config

    def compute_output_shape(self, input_shape):
        return input_shape
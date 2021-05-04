import tensorflow as tf
from tensorflow.keras.layers import Input,Dense, Lambda
from tensorflow.keras import Model


def make_model(num_classes, input_shape=(532,299,3), last_activation="linear"):
    model_xception = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=input_shape,
        pooling='avg'
    )
    input = Input(shape=input_shape)
    model_inception_cut = model_xception(input)
    output = Dense(num_classes,activation=last_activation)(model_inception_cut)
    output = Lambda(lambda x: tf.keras.backend.stack([x], axis=1))(
        output)  # Keras requiert le même nombre de dimensions pour y_true et y_pred (cf loss_mse)
    model = Model(inputs=input, outputs=output)
    return model
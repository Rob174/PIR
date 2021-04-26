import tensorflow as tf
from tensorflow.keras.layers import Input,MaxPooling2D, Flatten, Concatenate
from tensorflow.keras import Model


def spatialPyramidPooling(input):

    # Resolutions
    couche_resolution_fine = MaxPooling2D(pool_size=2,strides=2,padding="valid")(input)
    couche_resolution_medium = MaxPooling2D(pool_size=2, strides=2, padding="valid")(couche_resolution_fine)
    couche_resolution_large = MaxPooling2D(pool_size=2, strides=2, padding="valid")(couche_resolution_medium)

    flatten_couche_resolution_fine = Flatten()(couche_resolution_fine)
    flatten_couche_resolution_medium = Flatten()(couche_resolution_medium)
    flatten_couche_resolution_large = Flatten()(couche_resolution_large)

    output = Concatenate()([flatten_couche_resolution_fine,flatten_couche_resolution_medium,flatten_couche_resolution_large])
    return output
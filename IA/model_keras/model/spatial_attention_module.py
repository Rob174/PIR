import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Activation, Multiply, Lambda

def spatial_attention_module(input,nb_filtres_couche_input):
    """ Expliqué ici https://youtu.be/bDK9NRF20To?t=1643"""
    x = Conv2D(filters=nb_filtres_couche_input,kernel_size=2,padding='same')(input)
    x = Activation(activation='sigmoid')(x)
    return Multiply()([input,x])
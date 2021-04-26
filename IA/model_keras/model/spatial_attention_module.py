import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Activation, Multiply

def spatial_attention_module(input,nb_filtres_couche_input):
    x = Conv2D(filters=nb_filtres_couche_input,kernel_size=2,padding='same')
    x = Activation(activation='sigmoid')(x)
    return Multiply()([input,x])
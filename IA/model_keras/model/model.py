# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from IA.improved_graph.src.layers.base_layers import Input,Conv2D,BatchNormalization,Activation,SeparableConv2D,\
    MaxPooling2D,Add,GlobalAveragePooling2D,Dropout,Dense
from IA.improved_graph.src.layers.node_model import *


'''cette fonction permet de créer le modéle(architecture de l'IA)'''
'''couche par ordre input, couche de convolution(filtre, noyau,stride(pas de déplacement), padding(),batchNormalisation(ameliorer la convergence )'''
import tensorflow as tf
def make_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Entry block
    #preprocess the data
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    #fonction d'activation
    x = Activation(activation="relu")(x)

    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = Activation(activation="relu")(x)
        #réduit le nombre de parametre (matrice multipliée au lieu de matrice normale)
        x = SeparableConv2D(filters=size, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation(activation="relu")(x)
        x = SeparableConv2D(filters=size, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)

        #couche de pooling
        x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters=size, kernel_size=1, strides=2, padding="same")(previous_block_activation)
        x = Add()([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = SeparableConv2D(filters=1024, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)

    '''moyenne de toute les valeurs au lieu du max'''
    x = GlobalAveragePooling2D()(x)

    '''on met 50% des pixels en blanc pour eviter que le réseau se base sur les memes pixels (couche de régularisation)'''
    x = Dropout(rate=0.5)(x)
    #couche fully connected layer
    outputs = Dense(units=num_classes, activation="softmax")(x)
    return Model([inputs], [outputs],name="keras_model")

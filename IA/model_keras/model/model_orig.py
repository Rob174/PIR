# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from tensorflow.keras.layers import Dense,Conv2D,GlobalAveragePooling2D,Input,BatchNormalization,Activation,SeparableConv2D,Add,MaxPooling2D,Dropout,Flatten
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.regularizers as regularizers

from IA.model_keras.model.spatial_attention_module import spatial_attention_module
from IA.model_keras.model.spatial_pyramid_pooling import spatialPyramidPooling

'''cette fonction permet de créer le modéle(architecture de l'IA)'''
'''couche par ordre input, couche de convolution(filtre, noyau,stride(pas de déplacement), padding(),batchNormalisation(ameliorer la convergence )'''


def make_model(input_shape, num_classes, last_activation="linear", nb_modules=4, reduction_layer="globalavgpool",
               dropout_rate=0.5, spatial_attention="n"):
    inputs = Input(shape=input_shape)

    # Entry block
    # preprocess the data
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding="same",use_bias=False)(inputs)
    x = BatchNormalization()(x)
    # fonction d'activation
    x = Activation(activation="relu")(x)

    x = Conv2D(filters=64, kernel_size=3, padding="same",use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)

    previous_block_activation = x  # Set aside residual
    liste_filtres = [128, 256, 512, 728, 1024]
    liste_filtres = liste_filtres[:nb_modules + 1]
    for size in liste_filtres[:-1]:
        x = Activation(activation="relu")(x)
        # réduit le nombre de parametre (matrice multipliée au lieu de matrice normale)
        x = SeparableConv2D(filters=size, kernel_size=3, padding="same",use_bias=False)(x)
        x = BatchNormalization()(x)

        x = Activation(activation="relu")(x)
        x = SeparableConv2D(filters=size, kernel_size=3, padding="same",use_bias=False)(x)
        x = BatchNormalization()(x)

        # couche de pooling
        x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters=size, kernel_size=1, strides=2, padding="same",use_bias=False)(previous_block_activation)
        x = Add()([x, residual])  # Add back residual
        if spatial_attention == "y":
            x = spatial_attention_module(x,nb_filtres_couche_input=size*2)
        previous_block_activation = x  # Set aside next residual

    x = SeparableConv2D(filters=liste_filtres[-1], kernel_size=3, padding="same",use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)

    if reduction_layer == "globalavgpool":
        '''moyenne de toute les valeurs au lieu du max'''
        x = GlobalAveragePooling2D()(x)
    elif reduction_layer == "spp":
        x = spatialPyramidPooling(x)
    elif reduction_layer == "flatten":
        x = Flatten()(x)
    else:
        raise Exception("reduction layer option not supported %s" % reduction_layer)

    '''on met 50% des pixels en blanc pour eviter que le réseau se base sur les memes pixels (couche de régularisation)'''
    x = Dropout(rate=dropout_rate)(x)
    # couche fully connected layer
    outputs = Dense(units=num_classes, activation=last_activation)(x)
    outputs = tf.keras.backend.stack([outputs],axis=0) # Keras requiert le même nombre de dimensions pour y_true et y_pred (cf loss_mse)
    model =  Model([inputs], [outputs], name="keras_model")
    print(model.summary())
    return model


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.utils import plot_model


# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv):
    # add convolutional layers
    for _ in range(n_conv):
        layer_in = Conv3D(n_filters, (3, 3, ), padding='same', activation='relu')(layer_in)
    # add max pooling layer
    layer_in = MaxPooling3D((2, 2, 2), strides=(2, 2, 1))(layer_in)
    return layer_in


# define model input
visible = Input(shape=(112, 112, 16, 3))
# add x vgg modules
layer = vgg_block(visible, 16, 1)
layer = vgg_block(layer, 128, 1)
layer = vgg_block(layer, 128, 1)
layer = vgg_block(layer, 256, 1)
layer = vgg_block(layer, 256, 1)

# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file='frontmovingobject3D_test1.png')
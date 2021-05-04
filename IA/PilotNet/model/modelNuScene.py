import os
import sys

# Permet de lancer le script python directement et de reconnaitre les modules (IA.PilotNet.....) impérativement à mettre avant les from IA. ....

chemin_fichier = os.path.realpath(__file__).split("/")
os.chdir("/".join(chemin_fichier[:-2]))
print("/".join(chemin_fichier[:-3]))
print("/".join(chemin_fichier[:-2]))
print("/".join(chemin_fichier[:-2] + ["model"]))
sys.path.append("/".join(chemin_fichier[:-3]))
sys.path.append("/".join(chemin_fichier[:-4]))
sys.path.append("/".join(chemin_fichier[:-3] + ["improved_graph", "src", "layers"]))
# sys.path.append("/".join(chemin_fichier[:-2]))
# sys.path.append("/".join(chemin_fichier[:-2] + ["model"]))



from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten, BatchNormalization
from IA.PilotNet.data.NuSceneDB import NuSceneDB

def createNuSceneModel():
    input = Input(shape=(66, 200, 3))  # width, height, channel: 3 = RGB

    # Normalized input planes 3@66x200
    normalizedInput = BatchNormalization()(input)

    # Convolutional feature map 24@31x98 with a 5x5 kernel and a 2x2 stride
    layer = Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu')(normalizedInput)

    # Convolutional feature map 36@14x47 with a 5x5 kernel and a 2x2 stride
    layer = Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu')(layer)

    # Convolutional feature map 48@5x22 with a 5x5 kernel and a 2x2 stride
    layer = Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu')(layer)

    # Convolutional feature map 64@3x20 with a 3x3 kernel and no stride
    layer = Conv2D(64, (3, 3), padding='valid', activation='relu')(layer)

    # Convolutional feature map 64@1x18 with a 3x3 kernel and no stride
    layer = Conv2D(64, (3, 3), padding='valid', activation='relu')(layer)

    # Flatten to 1164 neurons
    flattenLayer = Flatten()(layer)

    # 100 neurons Fully-connected layer
    fullyConnectedLayer = Dense(units=100)(flattenLayer)

    # 50 neurons Fully-connected layer
    fullyConnectedLayer = Dense(units=50)(fullyConnectedLayer)

    # 10 neurons Fully-connected layer
    fullyConnectedLayer = Dense(units=10)(fullyConnectedLayer)

    # Length of labels array (=23) neurons Fully-connected layer
    fullyConnectedLayer = Dense(units=len(NuSceneDB.labels))(fullyConnectedLayer)

    # Output: vehicle control
    model = Model(inputs=[input], outputs=[fullyConnectedLayer])

    return model
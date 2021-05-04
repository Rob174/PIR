import os
import sys
import tensorflow as tf

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


from IA.PilotNet.data.NuSceneDB import NuSceneDB
from IA.PilotNet.model.modelOrig import createPilotNetModel
from IA.PilotNet.model.modelNuScene import createNuSceneModel
from IA.model_keras.FolderInfos import FolderInfos
from IA.model_keras.callbacks.EvalCallback import EvalCallback
from IA.model_keras.plot_graph.src.analyser.analyse import plot_model
from IA.PilotNet.markdown_summary.markdown_summary import create_summary
from PIL import Image
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


# images of the road ahead and steering angles in random order
BATCH_SIZE = 128
dataset = NuSceneDB(BATCH_SIZE)

dataset_tr = tf.data.Dataset.from_generator(dataset.train_batch_generator,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([None, None, None, None]),
                                                           tf.TensorShape([None, None])))\
    .prefetch(tf.data.experimental.AUTOTUNE).repeat(30)

dataset_tr_eval = tf.data.Dataset.from_generator(dataset.train_batch_generator,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([None, None, None, None]),
                                                           tf.TensorShape([None, None])))\
    .prefetch(tf.data.experimental.AUTOTUNE).repeat(30)
dataset_valid = tf.data.Dataset.from_generator(dataset.val_batch_generator,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([None, None, None, None]),
                                                           tf.TensorShape([None, None])))\
    .prefetch(tf.data.experimental.AUTOTUNE).repeat()  # Sans argument : répète à l'infini

FolderInfos.init(subdir="nuscene_pilotnet")

logdir = FolderInfos.base_folder
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

with tf.device('/GPU:' + "0"):
    # Create a model based on original (PilotNet) architecture
    pilotNetModel = createPilotNetModel()

    # Import weights for the original model
    # Link weights to the PilotNet model
    pilotNetModel.load_weights("/home/smaniott/PIR/PIR/data/pilotnet/2021-04-22_20h45min29s_/2021-04-22_20h45min29s_model.h5")

    # Freeze base model
    pilotNetModel.trainable = False

    # (Cut) Select last wanted layer of the original model (0 is the input layer, 11 is the steering_angle output layer)
    lastWantedLayerIndex = 9  # or -3 for the second last
    lastWantedLayer = pilotNetModel.layers[lastWantedLayerIndex]

    # Add layers to match the outputs (NuSceneDB.labels array)
    # Length of labels array (=23) neurons Fully-connected layer
    fullyConnectedLayer = Dense(units=len(NuSceneDB.labels))(lastWantedLayer.output)

    # Build the new model based on the original input and the previously added layers
    model = Model(inputs=[pilotNetModel.input], outputs=[fullyConnectedLayer])

    # Compile the model with given hyper parameters
    print(model.summary())
    model.compile(optimizer="adam", loss="MSE", metrics="accuracy")

name = FolderInfos.base_filename + "model.dot"
name_png = FolderInfos.base_filename + "model.png"
plot_model(model, output_path=name)
with file_writer.as_default():
    image = np.array(Image.open(name_png))
    tf.summary.image(f"Modele", np.stack((image,), axis=0), step=0)
    file_writer.flush()
    callbacks = None  # Pour le debug
    # """
    callbacks = [
        EvalCallback(file_writer, dataset_tr_eval, dataset.batch_size, ["loss_MSE"], type="tr"),
        EvalCallback(file_writer, dataset_valid, dataset.batch_size, ["loss_MSE"], type="valid",
                     eval_rate=dataset.batch_size * 5)
    ]
    # """

create_summary(writer=file_writer, optimizer_name="adam", optimizer_parameters={"lr":1e-3,"epsilon":1e-7}, loss="MSE",
               metriques_utilisees=[],
               but_essai="Test du nouveau modèle basé sur les poids de PilotNet/driving_dataset",
               informations_additionnelles="Couches enlevées : output à 1 neuronne et FC3",
               id=FolderInfos.id,dataset_name="driving_dataset from PilotNet",
               taille_x_img=1600,
               taille_y_img=900,batch_size=dataset.batch_size,
               nb_img_tot=dataset.dataset_size,nb_epochs=15,nb_tr_img=dataset.training_dataset_size) # Calculé d'après le fichier d'annotations
with tf.device('/GPU:' + "0"):
    print("===================================")
    print("Starting training")
    print("===================================")
    model.fit(dataset_tr, callbacks=callbacks)
    print("===================================")
    print("Training has ended")
    print("===================================")

    model.save(FolderInfos.base_filename+"model.h5")
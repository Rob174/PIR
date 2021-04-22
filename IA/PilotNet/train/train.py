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



from IA.PilotNet.data.ImageSteeringDB import ImageSteeringDB
from IA.PilotNet.model.modelOrig import create
from IA.model_keras.FolderInfos import FolderInfos
from IA.model_keras.callbacks.EvalCallback import EvalCallback
from IA.model_keras.plot_graph.src.analyser.analyse import plot_model
from IA.PilotNet.markdown_summary.markdown_summary import create_summary
from PIL import Image
import numpy as np


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass



# images of the road ahead and steering angles in random order
BATCH_SIZE = 128
dataset = ImageSteeringDB(BATCH_SIZE)

dataset_tr = tf.data.Dataset.from_generator(dataset.train_batch_generator,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([None, None, None, None]),
                                                           tf.TensorShape([None])))\
    .prefetch(tf.data.experimental.AUTOTUNE).repeat(30)

dataset_tr_eval = tf.data.Dataset.from_generator(dataset.train_batch_generator,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([None, None, None, None]),
                                                           tf.TensorShape([None])))\
    .prefetch(tf.data.experimental.AUTOTUNE).repeat(30)
dataset_valid = tf.data.Dataset.from_generator(dataset.val_batch_generator,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([None, None, None, None]),
                                                           tf.TensorShape([None])))\
    .prefetch(tf.data.experimental.AUTOTUNE).repeat() # Sans argument : répète à l'infini

FolderInfos.init(subdir="pilotnet")


logdir = FolderInfos.base_folder
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

with tf.device('/GPU:' + "0"):
    model = create()
    model.compile(optimizer="adam",loss="MSE")

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
               but_essai="Test du modèle original de PilotNet",
               informations_additionnelles="",
               id=FolderInfos.id,dataset_name="driving_dataset from PilotNet",
               taille_x_img=455,
               taille_y_img=256,batch_size=dataset.batch_size,
               nb_img_tot=45407,nb_epochs=30,nb_tr_img=32325) # Calculé d'après le fichier d'annotations
with tf.device('/GPU:' + "0"):
    model.fit(dataset_tr, callbacks=callbacks)
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD

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
from IA.PilotNet.analyse.matrices_confusion import MakeConfusionMatrix
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

# print(physical_devices)
# tf.debugging.set_log_device_placement(True)

GPU = str(0)

BATCH_SIZE = 128
NB_EPOCHS = 1

PILOTNET_LAST_LAYER = 10  # from 10 to 7

learning_rate = 1e-3

optimizer_name = "adam"  # adam or sgd

if(optimizer_name == "adam"):
    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-7)
else:
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=False)

LOSS = "MSE"

APPROX = "int"  # none, round, int

# images of the road ahead and steering angles in random order
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

FolderInfos.init(subdir="nuscene_pilotnet_epoch_" + str(NB_EPOCHS))

logdir = FolderInfos.base_folder
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()


def approx_accuracy(modeApprox="none"):
    fct_approx = None
    if modeApprox == "none":
        fct_approx = lambda x: x
    elif modeApprox == "round":
        fct_approx = tf.math.round
    elif modeApprox == "int":
        fct_approx = tf.math.floor
    else:
        raise Exception("Unknown approximation function %s" % modeApprox)

    def approx_accuracy_round(y_true, y_pred):
        print("\n")
        print("==========================================================")
        print("CALLING approx_accuracy_round with :")
        print("y_true = " + str(y_true))
        print("y_pred = " + str(y_pred))
        print("//////////////////////////////////////////////////////////")
        nb_classes = len(NuSceneDB.labels)

        # Extracting tensors' information
        y_pred_extracted = tf.slice(y_pred, [0, 0], size=[dataset.batch_size, nb_classes])
        y_pred_extracted = tf.reshape(y_pred_extracted, [dataset.batch_size, nb_classes])

        y_true_label = tf.slice(y_true, [0, 0], size=[dataset.batch_size, nb_classes])
        y_true_label = tf.reshape(y_true_label, [dataset.batch_size, nb_classes])

        print("y_true_label = " + str(y_true_label))
        print("y_pred_extracted = " + str(y_pred_extracted))
        print("Comparison: " + str(tf.equal(y_true_label, fct_approx(y_pred_extracted))))
        print("==========================================================")



        return tf.reduce_mean(tf.cast(tf.equal(y_true_label, fct_approx(y_pred_extracted)), dtype=tf.float32))

    return approx_accuracy_round

def add_pilotnet_layers(layer_from_index, layer):
    """
    layer_from_index : int :
        *   7 < layer_from_index < 10 : add PilotNet layers from layer_from_index + 1 to 10, AND NuScene labels layer
        *   layer_from = 10 : only add NuScene labels layer
    """
    if layer_from_index == 7:
        # 100 neurons Fully-connected layer
        layer = Dense(units=100)(layer.output)
        # 50 neurons Fully-connected layer
        layer = Dense(units=50)(layer)
        # Length of labels array (=23) neurons Fully-connected layer
        layer = Dense(units=len(NuSceneDB.labels))(layer)

    if layer_from_index == 8:
        # 50 neurons Fully-connected layer
        layer = Dense(units=50)(layer.output)
        # Length of labels array (=23) neurons Fully-connected layer
        layer = Dense(units=len(NuSceneDB.labels))(layer)

    if layer_from_index == 9:
        # Length of labels array (=23) neurons Fully-connected layer
        layer = Dense(units=len(NuSceneDB.labels))(layer.output)

    if layer_from_index == 10:
        # Length of labels array (=23) neurons Fully-connected layer
        layer = Dense(units=len(NuSceneDB.labels))(layer.output)

    return layer


with tf.device('/GPU:' + GPU):
    # Create a model based on original (PilotNet) architecture
    pilotNetModel = createPilotNetModel()

    # Import weights for the original model
    # Link weights to the PilotNet model
    pilotNetModel.load_weights("/home/smaniott/PIR/PIR/data/pilotnet/2021-04-22_20h45min29s_/2021-04-22_20h45min29s_model.h5")

    # Freeze base model
    pilotNetModel.trainable = False

    # (Cut) Select last wanted layer of the original model (0 is the input layer, 11 is the steering_angle output layer)
    lastWantedLayerIndex = PILOTNET_LAST_LAYER  # 10 or -2 for the second last
    lastWantedLayer = pilotNetModel.layers[lastWantedLayerIndex]

    # Add layers to match the outputs (NuSceneDB.labels array)
    fullyConnectedLayer = add_pilotnet_layers(lastWantedLayerIndex, lastWantedLayer)

    # Build the new model based on the original input and the previously added layers
    model = Model(inputs=[pilotNetModel.input], outputs=[fullyConnectedLayer])

    # Compile the model with given hyper parameters
    print(model.summary())
    model.compile(optimizer=optimizer_name, loss=LOSS, metrics=[approx_accuracy(APPROX)])

name = FolderInfos.base_filename + "model.dot"
name_png = FolderInfos.base_filename + "model.png"
plot_model(model, output_path=name)
with file_writer.as_default():
    image = np.array(Image.open(name_png))
    tf.summary.image(f"Model", np.stack((image,), axis=0), step=0)
    file_writer.flush()
    callbacks = None  # Pour le debug
    # """
    callbacks = [
        EvalCallback(file_writer, dataset_tr_eval, dataset.batch_size, ["loss_" + LOSS, "percentage_error"], type="tr"),
        EvalCallback(file_writer, dataset_valid, dataset.batch_size, ["loss_" + LOSS, "percentage_error"], type="valid",
                     eval_rate=dataset.batch_size * 5)
    ]
    # """

optimizer_parameters = {"lr":optimizer.lr.numpy(),"epsilon":optimizer.epsilon} if optimizer_name == "adam" else {"lr":optimizer.lr.numpy(),"momentum":optimizer.momentum.numpy()}

create_summary(writer=file_writer, optimizer_name=optimizer_name, optimizer_parameters=optimizer_parameters, loss=LOSS,
               metriques_utilisees=["Prediction error percentage"],
               but_essai="Test of a new model based on  PilotNet/driving_dataset weights",
               informations_additionnelles="Last wanted layer of the initial PilotNet model: " + str(lastWantedLayerIndex) + "\n Used approx : " + APPROX,
               id=FolderInfos.id,dataset_name="NuScene",
               taille_x_img=1600,
               taille_y_img=900,batch_size=dataset.batch_size,
               nb_img_tot=dataset.dataset_size,nb_epochs=NB_EPOCHS,nb_tr_img=dataset.training_dataset_size)
with tf.device('/GPU:' + GPU):
    print("===================================")
    print("Starting training")
    print("===================================")
    model.fit(dataset_tr, callbacks=callbacks, epochs=NB_EPOCHS)
    print("===================================")
    print("Training has ended")
    print("===================================")

    model.save(FolderInfos.base_filename+"model.h5")
    print("\n")
    print("===================================")
    print("Model has been saved in : " + FolderInfos.base_filename + "model.h5")
    print("===================================")

# Evaluation finale
dataset_full = tf.data.Dataset.from_generator(dataset.getNextBatchFullDataset,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape([None, None, None, None]),
                                                             tf.TensorShape([None, None]))) \
    .prefetch(tf.data.experimental.AUTOTUNE)

with tf.device('/GPU:' + GPU):
    MakeConfusionMatrix(model, dataset_full,
                        len(NuSceneDB.labels), NuSceneDB.labels,
                        summary_writer=file_writer, mode_approx=APPROX)()
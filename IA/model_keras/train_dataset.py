import argparse
import os
import sys
chemin_fichier = os.path.realpath(__file__).split("/")
sys.path.append("/".join(chemin_fichier[:-2]))
sys.path.append("/".join(chemin_fichier[:-3]))
print("/".join(chemin_fichier[:-2]+["improved_graph"]))
sys.path.append("/".join(chemin_fichier[:-2]+["improved_graph","src","layers"]))
from IA.improved_graph.src.layers.node_model import *
from IA.model_keras.data.generate_data import Nuscene_dataset
from model.model import make_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib
from tensorflow.keras.optimizers import Adam

matplotlib.use('Agg')

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

parser = argparse.ArgumentParser()

parser.add_argument('-bs', dest='batch_size', default=10, type=int,
                    help="[Optionnel] Indique le nombre d'images par batch")
parser.add_argument('-gpu', dest='gpu_selected', default="0", type=str,
                    help="[Optionnel] Indique la gpu visible par le script tensorflow")
parser.add_argument('-img_w', dest='image_width', default=400, type=int,
                    help="[Optionnel] Indique la gpu visible par le script tensorflow")
args = parser.parse_args()


dataset = Nuscene_dataset(img_width=args.image_width)
dataset.batch_size = args.batch_size
liste_lossTr = []
liste_accuracyTr = []
liste_lossValid = []
liste_accuracyValid = []
Lcoordx_tr = []
Lcoordx_valid = []

accur_step = 5
with tf.device('/GPU:'+args.gpu_selected):
    model = make_model((dataset.image_shape[1], dataset.image_shape[0], 3),
                       num_classes=len(dataset.correspondances_classes.keys()))
    model.keras_layer.compile(optimizer=Adam(learning_rate=1e-3, epsilon=1e-7), loss="MSE", metrics=["accuracy"])
from time import strftime, gmtime
import os
class FolderInfos:
    base_folder = None
    base_filename = None
    @staticmethod
    def init():
        id = strftime("%Y-%m-%d_%Hh%Mmin%Ss", gmtime())
        FolderInfos.base_folder= "/".join(os.path.realpath(__file__).split("/")[:-2]+["data"])+id+"/"
        FolderInfos.base_filename = FolderInfos.base_folder + id
        os.mkdir(FolderInfos.base_folder)
FolderInfos.init()
model.save(FolderInfos.base_filename+"graph.dot")
iteratorValid = dataset.getNextBatchValid()
compteur = 0


def plot():
    fig, axe_error = plt.subplots()
    loss_axe = axe_error.twinx()
    loss_axe.plot(
        np.array(Lcoordx_tr) * dataset.batch_size,
        liste_lossTr, color="r", label="lossTr")
    loss_axe.plot(np.array(Lcoordx_valid) * dataset.batch_size, liste_lossValid, color="orange", label="lossValid")
    axe_error.plot(np.array(Lcoordx_tr) * dataset.batch_size, 100 * (1 - np.array(liste_accuracyTr)), color="g",
                   label="tr_error")
    axe_error.plot(np.array(Lcoordx_valid) * dataset.batch_size, 100 * (1 - np.array(liste_accuracyValid)), color="b",
                   label="valid_error")
    axe_error.set_xlabel("Nombre d'itérations, d'images passées")
    axe_error.set_ylabel("Error (%)")
    loss_axe.set_ylabel("Loss (MSE)")
    fig.legend()
    plt.grid()
    plt.savefig("/home/rmoine/Documents/erreur_accuracy_batch_size_%d.png" % dataset.batch_size)
    plt.clf()
    plt.close(fig)


for epochs in range(1):
    iteratorTr = dataset.getNextBatchTr()
    while True:
        try:
            batchImg, batchLabel = next(iteratorTr)
            with tf.device('/GPU:'+args.gpu_selected):
                [loss, accuracy] = model.train_on_batch(batchImg, batchLabel)
            liste_lossTr.append(loss)
            liste_accuracyTr.append(accuracy)
            Lcoordx_tr.append(compteur)
            compteur = compteur + 1
            if compteur % accur_step == 0:
                batchImg, batchLabel = next(iteratorValid)
                with tf.device('/GPU:' + args.gpu_selected):
                    [loss, accuracy] = model.test_on_batch(batchImg, batchLabel)
                liste_lossValid.append(loss)
                liste_accuracyValid.append(accuracy)
                Lcoordx_valid.append(compteur)
                plot()
            if compteur == 100:
                break
        except StopIteration:
            print("Epoch %d done" % epochs)
            break

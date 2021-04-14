from data.generate_data import Nexet_dataset
from model.model import make_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib
from tensorflow.keras.optimizers import Adam
import argparse
parser = argparse.ArgumentParser()
parser.parse_args()

parser.add_argument('-bs', dest='batch_size', default=10, type=int,
                    help="[Optionnel] Indique le nombre d'images par batch")
matplotlib.use('Agg')

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
      tf.config.experimental.set_memory_growth(device, True)
    except:
      # Invalid device or cannot modify virtual devices once initialized.
      pass
dataset = Nexet_dataset()
dataset.batch_size = parser.batch_size
liste_lossTr=[]
liste_accuracyTr=[]
liste_lossValid=[]
liste_accuracyValid=[]
Lcoordx_tr = []
Lcoordx_valid = []

accur_step = 5
model = make_model((dataset.image_shape[1], dataset.image_shape[0],3), num_classes=len(dataset.correspondances_classes.keys()))
model.compile(optimizer=Adam(learning_rate=1e-3,epsilon=1e-1), loss="MSE", metrics=["accuracy"])
iteratorValid = dataset.getNextBatchValid()
compteur=0

def plot():
    fig, axe_error = plt.subplots()
    loss_axe = axe_error.twinx()
    loss_axe.plot(
        np.array(Lcoordx_tr)*dataset.batch_size,
        liste_lossTr, color="r", label="lossTr")
    loss_axe.plot(np.array(Lcoordx_valid)*dataset.batch_size, liste_lossValid,color="orange",label="lossValid")
    axe_error.plot(np.array(Lcoordx_tr)*dataset.batch_size, 100 * (1 - np.array(liste_accuracyTr)), color="g", label="tr_error")
    axe_error.plot(np.array(Lcoordx_valid)*dataset.batch_size,100*(1-np.array(liste_accuracyValid)),color="b",label="valid_error")
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
            [loss,accuracy]=model.train_on_batch(batchImg,batchLabel)
            liste_lossTr.append(loss)
            liste_accuracyTr.append(accuracy)
            Lcoordx_tr.append(compteur)
            compteur=compteur+1
            if compteur % accur_step == 0:
                batchImg, batchLabel = next(iteratorValid)
                [loss,accuracy]=model.test_on_batch(batchImg,batchLabel)
                liste_lossValid.append(loss)
                liste_accuracyValid.append(accuracy)
                Lcoordx_valid.append(compteur)
                plot()
            if compteur == 1080:
                break
        except StopIteration:
            print("Epoch %d done" % epochs)
            break
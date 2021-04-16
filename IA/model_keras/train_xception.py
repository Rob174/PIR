import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras import Model
import argparse
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow_core.python.keras.optimizers import SGD
import numpy as np

chemin_fichier = os.path.realpath(__file__).split("/")
sys.path.append("/".join(chemin_fichier[:-2]))
sys.path.append("/".join(chemin_fichier[:-3]))
print("/".join(chemin_fichier[:-2] + ["improved_graph"]))
sys.path.append("/".join(chemin_fichier[:-2] + ["improved_graph", "src", "layers"]))
from IA.model_keras.data.generate_data import Nuscene_dataset
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
parser.add_argument('-lr', dest='lr', default=1e-3, type=float,
                    help="[Optionnel] Indique la gpu visible par le script tensorflow")
parser.add_argument('-eps', dest='epsilon', default=1e-7, type=float,
                    help="[Optionnel] Indique la gpu visible par le script tensorflow")
parser.add_argument('-lastAct', dest='lastActivation', default="linear", type=str,
                    help="[Optionnel] Indique la gpu visible par le script tensorflow")
parser.add_argument('-approxAccur', dest='approximationAccuracy', default="none", type=str,
                    help="[Optionnel] Indique la gpu visible par le script tensorflow")
parser.add_argument('-nbMod', dest='nb_modules', default=4, type=int,
                    help="[Optionnel] Indique la gpu visible par le script tensorflow")
parser.add_argument('-redLayer', dest='reduction_layer', default="globalavgpool", type=str,
                    help="[Optionnel] Indique la gpu visible par le script tensorflow")
parser.add_argument('-dptRate', dest='dropout_rate', default="0.5", type=str,
                    help="[Optionnel] Indique la gpu visible par le script tensorflow")
parser.add_argument('-opti', dest='optimizer', default="adam", type=str,
                    help="[Optionnel] Optimisateur")
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


def approx_accuracy(modeApprox="none"):
    fct_approx = None
    if modeApprox == "none":
        fct_approx = lambda x: x
    elif args.approximationAccuracy == "round":
        fct_approx = tf.math.round
    elif args.approximationAccuracy == "int":
        fct_approx = tf.math.floor
    else:
        raise Exception("Unknown approximation function %s" % modeApprox)

    def approx_accuracy_round(y_true, y_pred):
        return categorical_accuracy(y_true, fct_approx(y_pred))

    return approx_accuracy_round

from time import strftime, gmtime
import os
class FolderInfos:
    base_folder = None
    base_filename = None

    @staticmethod
    def init():
        id = strftime("%Y-%m-%d_%Hh%Mmin%Ss", gmtime())
        FolderInfos.base_folder = "/".join(os.path.realpath(__file__).split("/")[:-3] + ["data/"]) + id + "/"
        FolderInfos.base_filename = FolderInfos.base_folder + id
        os.mkdir(FolderInfos.base_folder)


FolderInfos.init()

iteratorValid = dataset.getNextBatchValid()
compteur = 0


with tf.device('/GPU:' + "2"):
    model_xception = tf.keras.applications.Xception(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classifier_activation="softmax"
    )
    input = Input(shape=(299,299,3))
    model_inception_cut = Model(inputs=[model_xception.input],output=[model_xception.get_layer("avg_pool").output])(input)
    output = Dense(len(dataset.correspondances_classes.keys()))
    model = Model(inputs=[input],outputs=[output])
    model.compile(optimizer=SGD( learning_rate=0.045, momentum=0.9, nesterov=False), loss="MSE",
                  metrics=[approx_accuracy(args.approximationAccuracy)])

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
    plt.savefig(
        FolderInfos.base_filename + "erreur_accuracy_batch_size_%d_lastAct_%s_accurApprox_%s_nbMod_%d_dpt_%s_redLay_%s.png"
        % (dataset.batch_size, args.lastActivation, args.approximationAccuracy, args.nb_modules,
           args.dropout_rate, args.reduction_layer))
    plt.clf()
    plt.close(fig)

def adapt_image(image):
    padded_img = np.zeros((299,299,3))
    padded_img[:image.shape[0],:,:] = image
    return padded_img

for epochs in range(1):
    iteratorTr = dataset.getNextBatchTr()
    while True:
        try:
            batchImg, batchLabel = next(iteratorTr)
            with tf.device('/GPU:' + args.gpu_selected):
                [loss, accuracy] = model.keras_layer.train_on_batch(adapt_image(batchImg), batchLabel)
            liste_lossTr.append(loss)
            liste_accuracyTr.append(accuracy)
            Lcoordx_tr.append(compteur)
            compteur = compteur + 1
            if compteur % accur_step == 0:
                batchImg, batchLabel = next(iteratorValid)
                with tf.device('/GPU:' + args.gpu_selected):
                    [loss, accuracy] = model.keras_layer.test_on_batch(adapt_image(batchImg), batchLabel)
                liste_lossValid.append(loss)
                liste_accuracyValid.append(accuracy)
                Lcoordx_valid.append(compteur)
                plot()
            if compteur == 1080:
                break
        except StopIteration:
            print("Epoch %d done" % epochs)
            break
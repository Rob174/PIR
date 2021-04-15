import os
import sys


chemin_fichier = os.path.realpath(__file__).split("/")
sys.path.append("/".join(chemin_fichier[:-2]))
sys.path.append("/".join(chemin_fichier[:-3]))
print("/".join(chemin_fichier[:-2]+["improved_graph"]))
sys.path.append("/".join(chemin_fichier[:-2]+["improved_graph","src","layers"]))
from IA.model_keras.data.generate_data import Nuscene_dataset
from model.model import make_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from IA.model_keras.parser.parser import parse
from IA.model_keras.foldersInfos.FolderInfo import FolderInfos


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

args = parse()


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

FolderInfos.init()
model.save(FolderInfos.base_filename+"graph.dot")
iteratorValid = dataset.getNextBatchValid()
compteur = 0





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

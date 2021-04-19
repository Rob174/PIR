import os
import sys

from model_keras.callbacks.EvalCallback import EvalCallback

chemin_fichier = os.path.realpath(__file__).split("/")
sys.path.append("/".join(chemin_fichier[:-2]))
sys.path.append("/".join(chemin_fichier[:-3]))
print("/".join(chemin_fichier[:-2] + ["improved_graph"]))
sys.path.append("/".join(chemin_fichier[:-2] + ["improved_graph", "src", "layers"]))
from IA.model_keras.data.generate_data import Nuscene_dataset
from IA.model_keras.model.model_orig import make_model
from IA.model_keras.plot_graph.src.analyser.analyse import plot_model
from IA.model_keras.model.model_inception_top import make_model as make_model_inception_top
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.metrics import categorical_accuracy
from IA.model_keras.parser import parse

# faire le padding des images
matplotlib.use('Agg')

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


with tf.device('/GPU:' + args.gpu_selected):
    model = make_model((dataset.image_shape[1], dataset.image_shape[0], 3,),
                     num_classes=len(dataset.correspondances_classes.keys()),
                     last_activation=args.lastActivation,
                     nb_modules=args.nb_modules, reduction_layer=args.reduction_layer,
                     dropout_rate=float(args.dropout_rate))
    if args.optimizer == "adam":
        optimizer = Adam(learning_rate=args.lr, epsilon=args.epsilon)
    elif args.optimizer == "sgd":
        optimizer = SGD( learning_rate=0.045, momentum=0.9, nesterov=False)
    else:
        raise Exception("Optimizer %s not supported" % args.optimizer)
    model.compile(optimizer=optimizer, loss="MSE",
                              metrics=[approx_accuracy(args.approximationAccuracy)])


from IA.model_keras.FolderInfos import FolderInfos


FolderInfos.init()
plot_model(model,output_path=FolderInfos.base_filename + "_bs_%d_lastAct_%s_accurApprox_%s_nbMod_%d_dpt_%s_redLay_%s_graph.dot"
           % (dataset.batch_size, args.lastActivation, args.approximationAccuracy,
              args.nb_modules, args.dropout_rate, args.reduction_layer))
iteratorValid = dataset.getNextBatchValid()
compteur = 0


dataset_tr = tf.data.Dataset.from_generator(dataset.getNextBatchTr,output_types=(tf.float32,tf.float32))\
                    .prefetch(tf.data.experimental.AUTOTUNE)
dataset_valid = tf.data.Dataset.from_generator(dataset.getNextBatchValid,output_types=(tf.float32,tf.float32))\
                    .prefetch(tf.data.experimental.AUTOTUNE).repeat()

tb_callback = tf.keras.callbacks.TensorBoard(FolderInfos.base_folder)
model.fit(dataset_tr,callbacks=[
    EvalCallback(tb_callback,dataset_valid,dataset.batch_size,["loss_MSE","prct_error"],type="tr"),
    EvalCallback(tb_callback,dataset_valid,dataset.batch_size,["loss_MSE","prct_error"],type="valid",
                 eval_rate=dataset.batch_size*4)
])

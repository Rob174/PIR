import os
import sys

chemin_fichier = os.path.realpath(__file__).split("/")
sys.path.append("/".join(chemin_fichier[:-2]))
sys.path.append("/".join(chemin_fichier[:-3]))
print("/".join(chemin_fichier[:-2] + ["improved_graph"]))
sys.path.append("/".join(chemin_fichier[:-2] + ["improved_graph", "src", "layers"]))

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD

from IA.model_keras.data.generate_data import Nuscene_dataset
from IA.model_keras.model.model_orig import make_model
from IA.model_keras.plot_graph.src.analyser.analyse import plot_model
from tensorflow.keras.metrics import categorical_accuracy
from IA.model_keras.parser import parse
from IA.model_keras.markdown_summary.markdown_summary import create_summary
from IA.model_keras.callbacks.EvalCallback import EvalCallback

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

args = parse()

dataset = Nuscene_dataset(img_width=args.image_width,limit_nb_tr=100)
dataset.batch_size = args.batch_size


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
        optimizer_params = {"learning_rate": args.lr, "epsilon": args.epsilon}
        optimizer = Adam(learning_rate=args.lr, epsilon=args.epsilon)
    elif args.optimizer == "sgd":
        optimizer_params = {"learning_rate": args.lr}
        optimizer = SGD(learning_rate=0.045, momentum=0.9, nesterov=False)
    else:
        raise Exception("Optimizer %s not supported" % args.optimizer)
    model.compile(optimizer=optimizer, loss="MSE",
                  metrics=[approx_accuracy(args.approximationAccuracy)])

from IA.model_keras.FolderInfos import FolderInfos

FolderInfos.init()
plot_model(model, output_path=FolderInfos.base_filename + "model.dot")

logdir = FolderInfos.base_folder
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

## Résumé des paramètres d'entrainement dans un markdown afficher dans le tensorboard
create_summary(file_writer, args.optimizer, optimizer_params, "MSE", [f"pourcent d'erreur de" +
                                                                      f" prediction en appliquant la fonction" +
                                                                      f" {args.approximationAccuracy} " +
                                                                      f"(none = identity) aux prédictions" +
                                                                      f" au préalable"],
               but_essai="Test du framework", informations_additionnelles="",
               model_img_path=FolderInfos.base_filename + "model.png")

dataset_tr = tf.data.Dataset.from_generator(dataset.getNextBatchTr, output_types=(tf.float32, tf.float32),
                                            output_shapes=(
                                            tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None]))) \
    .prefetch(tf.data.experimental.AUTOTUNE)
dataset_tr_eval = tf.data.Dataset.from_generator(dataset.getNextBatchTr, output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape([None, None, None, None]),
                                                                tf.TensorShape([None, None]))) \
    .prefetch(tf.data.experimental.AUTOTUNE)
dataset_valid = tf.data.Dataset.from_generator(dataset.getNextBatchValid, output_types=(tf.float32, tf.float32),
                                               output_shapes=(
                                               tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None]))) \
    .prefetch(tf.data.experimental.AUTOTUNE).repeat()

with tf.device('/GPU:' + args.gpu_selected):
    model.fit(dataset_tr, callbacks=[
        EvalCallback(file_writer, dataset_tr_eval, dataset.batch_size, ["loss_MSE", "prct_error"], type="tr"),
        EvalCallback(file_writer, dataset_valid, dataset.batch_size, ["loss_MSE", "prct_error"], type="valid",
                     eval_rate=dataset.batch_size * 5)
    ])

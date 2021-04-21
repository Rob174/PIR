import json
import os
import sys

chemin_fichier = os.path.realpath(__file__).split("/")
sys.path.append("/".join(chemin_fichier[:-2]))
sys.path.append("/".join(chemin_fichier[:-3]))
print("/".join(chemin_fichier[:-2] + ["improved_graph"]))
sys.path.append("/".join(chemin_fichier[:-2] + ["improved_graph", "src", "layers"]))

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD

from IA.model_keras.data.Nuscene_dataset import Nuscene_dataset
from IA.model_keras.model.model_orig import make_model
from IA.model_keras.plot_graph.src.analyser.analyse import plot_model
from tensorflow.keras.metrics import categorical_accuracy
from IA.model_keras.parser import parse
from IA.model_keras.markdown_summary.markdown_summary import create_summary
from IA.model_keras.callbacks.EvalCallback import EvalCallback
from IA.model_keras.FolderInfos import FolderInfos


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

args = parse()

FolderInfos.init()

dataset = Nuscene_dataset(img_width=args.image_width,limit_nb_tr=args.nb_images,taille_mini_px=args.taille_mini_obj_px,
                          batch_size=args.batch_size,data_folder=FolderInfos.data_folder)


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
        nb_classes = len(Nuscene_dataset.correspondances_classes.keys())
        # Extraction des informations des tenseurs
        y_pred_extracted = tf.slice(y_pred,[0,0,0],size=[1,dataset.batch_size,nb_classes])
        y_pred_extracted = tf.reshape(y_pred_extracted,[dataset.batch_size,nb_classes])

        y_true_label = tf.slice(y_true,[0,0,0],size=[dataset.batch_size,1,nb_classes])
        y_true_label = tf.reshape(y_true_label,[dataset.batch_size,nb_classes])

        return categorical_accuracy(y_true_label, fct_approx(y_pred_extracted))

    return approx_accuracy_round

def loss_mse(y_true,y_pred):
    nb_classes = len(Nuscene_dataset.correspondances_classes.keys())
    # Extraction des informations des tenseurs
    y_pred_extracted = tf.slice(y_pred,[0,0,0],size=[1,dataset.batch_size,nb_classes])
    y_pred_extracted = tf.reshape(y_pred_extracted,[dataset.batch_size,nb_classes])

    y_true_label = tf.slice(y_true,[0,0,0],size=[dataset.batch_size,1,nb_classes])
    y_true_label = tf.reshape(y_true_label,[dataset.batch_size,nb_classes])

    y_true_poids = tf.slice(y_true, [0, 1, 0], size=[dataset.batch_size, 1, nb_classes])
    y_true_poids = tf.reshape(y_true_poids,[dataset.batch_size,nb_classes])

    return tf.math.reduce_mean(tf.pow(y_true_label-y_pred_extracted,2)*y_true_poids)

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
    model.compile(optimizer=optimizer, loss=loss_mse,
                  metrics=[approx_accuracy(args.approximationAccuracy)])

plot_model(model, output_path=FolderInfos.base_filename + "model.dot")

logdir = FolderInfos.base_folder
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

texte_poids_par_classe = "\nPondération de chaque image suivant le nombre d'apparition de chaque classe du vecteur "
texte_poids_par_classe_eff= "\nPondération de chaque label suivant l'effectif d'apparition de chaque classe"
informations_additionnelles = "\nUtilisation d'une métrique custom pour corriger cela\n"+ "Normalisation des images par 255 avant passage dans le réseau"
informations_additionnelles += f"Garde un objet si une fois l'image redimensionnée il fait plus de {dataset.taille_mini_px} pixels (avec sa dimension minimale)"
if args.classes_weights == "class":
    informations_additionnelles += texte_poids_par_classe
elif args.classes_weigths == "classEff":
    informations_additionnelles += texte_poids_par_classe_eff
## Résumé des paramètres d'entrainement dans un markdown afficher dans le tensorboard
create_summary(file_writer, args.optimizer, optimizer_params, "MSE", [f"pourcent d'erreur de" +
                                                                      f" prediction en appliquant la fonction" +
                                                                      f" {args.approximationAccuracy} " +
                                                                      f"(none = identity) aux prédictions" +
                                                                      f" au préalable"],
               but_essai="Tester le modèle avec des images beaucoup plus petites",
               informations_additionnelles=informations_additionnelles,
               id=FolderInfos.id,dataset_name="Nuscene",taille_x_img_redim=dataset.image_shape[0],
               taille_y_img_redim=dataset.image_shape[1])

tr_generator_fct = None
valid_generator_fct = None

if args.classes_weights == "class":
    tr_generator_fct = lambda x: dataset.getNextBatchTr(with_weights="class")
    valid_generator_fct = lambda x: dataset.getNextBatchValid(with_weights="class")
elif args.classes_weigths == "classEff":
    tr_generator_fct = lambda x: dataset.getNextBatchTr(with_weights="classEff")
    valid_generator_fct = lambda x: dataset.getNextBatchValid(with_weights="classEff")


dataset_tr = tf.data.Dataset.from_generator(dataset.getNextBatchTr,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([None, None, None, None]),
                                                           tf.TensorShape([None, None, None])))\
    .prefetch(tf.data.experimental.AUTOTUNE)
dataset_tr_eval = tf.data.Dataset.from_generator(dataset.getNextBatchTr,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape([None, None, None, None]),
                                                                tf.TensorShape([None, None, None]))) \
    .prefetch(tf.data.experimental.AUTOTUNE)
dataset_valid = tf.data.Dataset.from_generator(dataset.getNextBatchValid,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=(tf.TensorShape([None, None, None, None]),
                                                              tf.TensorShape([None, None, None]))) \
    .prefetch(tf.data.experimental.AUTOTUNE).repeat()

path_weights = "/".join(FolderInfos.base_folder.split("/")[:-2])\
               +"/2021-04-19_12h06min43s_class_distribution_nuscene/"\
               +"2021-04-19_12h06min43s_class_distribution_stat_nb_elem_per_class.json"

callbacks = None # Pour le debug
"""
callbacks = [
        EvalCallback(file_writer, dataset_tr_eval, dataset.batch_size, ["loss_MSE", "prct_error"], type="tr"),
        EvalCallback(file_writer, dataset_valid, dataset.batch_size, ["loss_MSE", "prct_error"], type="valid",
                     eval_rate=dataset.batch_size * 5)
    ]
# """
with tf.device('/GPU:' + args.gpu_selected):
    model.fit(dataset_tr, callbacks=callbacks)

import os
import sys
import numpy as np
from PIL import Image


chemin_fichier = os.path.realpath(__file__).split("/")
sys.path.append("/".join(chemin_fichier[:-2]))
sys.path.append("/".join(chemin_fichier[:-3]))
sys.path.append("/".join(chemin_fichier[:-2] + ["improved_graph", "src", "layers"]))

from IA.enet.models.model_transfert import create
from IA.model_keras.parsers.parser0 import Parser0
args = Parser0()()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_selected
args.gpu_selected="0"
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy

from IA.enet.data.Nuscene_dataset_segmentation import Nuscene_dataset_segmentation
from IA.model_keras.plot_graph.src.analyser.analyse import plot_model
from tensorflow.keras.metrics import categorical_accuracy
from IA.model_keras.markdown_summary.markdown_summary import create_summary
from IA.model_keras.callbacks.EvalCallback import EvalCallback
from IA.model_keras.FolderInfos import FolderInfos
from IA.enet.analyse.matrice_confusion import MakeConfusionMatrixEnet
import matplotlib
matplotlib.use('Agg')
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


FolderInfos.init(subdir="enet")


logdir = FolderInfos.base_folder
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()


dataset = Nuscene_dataset_segmentation(img_width=532,limit_nb_tr=args.nb_images,seuils_threshold=[0.5,0.75,0.9],
                                       taille_mini_px=args.taille_mini_obj_px,
                                        batch_size=args.batch_size,with_weights="False",
                                        summary_writer=file_writer,augmentation=args.augmentation)

with tf.device('/GPU:' + args.gpu_selected):
    model = create(512,288,nb_classes=23) # TODO : predre multiples de 2
    if args.optimizer == "adam":
        optimizer_params = {"learning_rate": args.lr, "epsilon": args.epsilon}
        optimizer = Adam(learning_rate=args.lr, epsilon=args.epsilon)
    elif args.optimizer == "sgd":
        optimizer_params = {"learning_rate": args.lr}
        optimizer = SGD(learning_rate=args.lr, momentum=0.9, nesterov=False)
    else:
        raise Exception("Optimizer %s not supported" % args.optimizer)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                  metrics=[categorical_accuracy])

name = FolderInfos.base_filename + "model.dot"
name_png = FolderInfos.base_filename + "model.png"
plot_model(model, output_path=name)
with file_writer.as_default():
    image = np.array(Image.open(name_png))
    tf.summary.image(f"Modele",np.stack((image,),axis=0),step=0)
    file_writer.flush()

informations_additionnelles = "\n\nTransfert learning depuis les poids de pytorch tels que fournis dans le repo d'origine," + \
                              " coupe uniquement du dernier layer\n\n"
informations_additionnelles += "Images de 532px contre 200 pour les premiers essais\n\n"
informations_additionnelles += "Normalisation des images par 255 avant passage dans le réseau\n\n"
informations_additionnelles += f"Garde un objet si une fois l'image redimensionnée il fait plus de {dataset.taille_mini_px} pixels (avec sa dimension minimale)"

if args.augmentation == "t":
    informations_additionnelles += "\n\nAugmentations : \n"+"\n- ".join(list(map(
        lambda x:x.__name__+", paramètres : "+str(x.augm_params),dataset.augmentations)))

## Résumé des paramètres d'entrainement dans un markdown afficher dans le tensorboard
create_summary(writer=file_writer, optimizer_name=args.optimizer, optimizer_parameters=optimizer_params, loss="categorical_crossentropy",
               metriques_utilisees=[f"pourcent d'erreur de" +
                                    f" prediction"],
               but_essai="",
               informations_additionnelles=informations_additionnelles,
               id=FolderInfos.id,dataset_name="Nuscene",taille_x_img_redim=dataset.image_shape[0],
               taille_y_img_redim=dataset.image_shape[1],batch_size=dataset.batch_size,
               nb_img_tot=173959,nb_img_utilisees=args.nb_images,nb_epochs=args.nb_epochs)


dataset_tr = tf.data.Dataset.from_generator(dataset.getNextBatchTr,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([None, None, None, None]),
                                                           tf.TensorShape([None, None, None,None])))\
    .prefetch(tf.data.experimental.AUTOTUNE).repeat(args.nb_epochs)
dataset_tr_eval = tf.data.Dataset.from_generator(dataset.getNextBatchTr,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape([None, None, None, None]),
                                                                tf.TensorShape([None, None, None,None]))) \
    .prefetch(tf.data.experimental.AUTOTUNE).repeat(args.nb_epochs)
dataset_valid = tf.data.Dataset.from_generator(dataset.getNextBatchValid,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=(tf.TensorShape([None, None, None, None]),
                                                              tf.TensorShape([None, None, None,None]))) \
    .prefetch(tf.data.experimental.AUTOTUNE).repeat()

callbacks = None # Pour le debug
# """
callbacks = [
        EvalCallback(file_writer, dataset_tr_eval, dataset.batch_size, ["loss_categorical_crossentropy", "prct_error"], type="tr"),
        EvalCallback(file_writer, dataset_valid, dataset.batch_size, ["loss_categorical_crossentropy", "prct_error"], type="valid",
                     eval_rate=dataset.batch_size * 5)
    ]
# """

# with tf.device('/GPU:' + args.gpu_selected):
with tf.device('/GPU:'+args.gpu_selected):
    model.fit(dataset_tr, callbacks=callbacks)


# Evaluation finale
dataset_full = tf.data.Dataset.from_generator(dataset.getNextBatchFullDataset,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=(tf.TensorShape([None, None, None, None]),
                                                              tf.TensorShape([None, None, None,None]))) \
    .prefetch(tf.data.experimental.AUTOTUNE)

with tf.device('/GPU:'+args.gpu_selected):
    MakeConfusionMatrixEnet(model=model, dataset=dataset_full,seuils_threshold=dataset.seuils_threshold,
                            nb_classes=dataset.nb_classes,
                            correspondances_index_classes=dataset.correspondances_index_classes, mode_approx='none',
                            summary_writer=file_writer)()
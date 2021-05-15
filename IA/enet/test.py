import os
import sys
import numpy as np
from PIL import Image


chemin_fichier = os.path.realpath(__file__).split("/")
sys.path.append("/".join(chemin_fichier[:-2]))
sys.path.append("/".join(chemin_fichier[:-3]))
sys.path.append("/".join(chemin_fichier[:-2] + ["improved_graph", "src", "layers"]))

from IA.enet.data.Nuscene_dataset_segmentation import Nuscene_dataset_segmentation
import tensorflow as tf

dataset = Nuscene_dataset_segmentation(img_width=1600,limit_nb_tr=7500,
                                       taille_mini_px=0,
                                        batch_size=10,with_weights="False",
                                        summary_writer=None,augmentation="f")

dataset_tr = tf.data.Dataset.from_generator(dataset.getNextBatchTr,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([None, None, None, None]),
                                                           tf.TensorShape([None, None, None,None])))\
    .prefetch(tf.data.experimental.AUTOTUNE).repeat(1)

super(type(dataset),dataset).getLabels(0)
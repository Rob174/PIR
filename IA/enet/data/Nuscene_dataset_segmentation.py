import numpy as np

from IA.model_keras.FolderInfos import FolderInfos
from IA.model_keras.data.Nuscene_dataset import Nuscene_dataset
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from pandas.plotting import table
import seaborn as sns
import tensorflow as tf

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Nuscene_dataset_segmentation(Nuscene_dataset):

    def __init__(self, seuils_threshold=(0.5,0.75,0.90),*args, **kargs):
        self.seuils_threshold = seuils_threshold
        super(Nuscene_dataset_segmentation, self).__init__(*args, **kargs)
        self.correspondances_index_classes = {v:k for k,v in Nuscene_dataset.class_to_index.items()}
        self.get_labels_fct = self.getLabels

    def dataset_stats(self,summary_writer):
        if summary_writer is None:
            return
        print(f"{bcolors.OKBLUE}DONE !{bcolors.ENDC}")
        for dataset_name,dataset in zip(["tr","valid"],[self.dataset_tr,self.dataset_valid]):
            repartition_objets = {classe:{} for classe in self.class_to_index.keys()}
            for i in dataset:
                label = self.getLabels(i)
                for seuil_threshold in self.seuils_threshold:
                    for class_name,i_class in self.class_to_index.items():
                        _, img_threshold = cv2.threshold(label[ :, :, i_class],
                                                         seuil_threshold,
                                                         maxval=1.,
                                                         type=cv2.THRESH_BINARY)
                        img_threshold = (img_threshold * 255).astype(np.uint8)
                        nb_contours = str(len(cv2.findContours(img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]))
                        if nb_contours not in repartition_objets[class_name]:
                            repartition_objets[class_name][nb_contours] = 0
                        repartition_objets[class_name][nb_contours] += 1
            df: pd.DataFrame = pd.DataFrame(repartition_objets)
            df.to_csv(f"{FolderInfos.base_filename}_statistiques_{dataset_name}.csv")
            ax = sns.heatmap(df,annot=True,fmt='d', linewidths=.5)
            fig = ax.get_figure()
            # get image in numpy array (thanks to https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data = np.stack((data,), axis=0)

            tf.summary.image(f"dataset_{dataset_name}_stats", data, step=0)
            plt.savefig(f"{FolderInfos.base_filename}_stats_{dataset_name}")
            summary_writer.flush()
            plt.close()
        print(f"{bcolors.OKBLUE}END !{bcolors.ENDC}")

    def getLabels(self, index_image,num_batch=None,*args,**kargs):
        """
        Récupération et création des labels d'index index_image dans le fichier json du dataset Nuscene
        :param index_image: index de l'image dans le fichier json des annotations de Nuscene
        :return: np.array de shape (img_width, img_height, #classes) contenant la probabilité que chaque pixel appartienne à chaque classe
        """
        dico_categorie_image = self.content_dataset[index_image]["categories"]
        label = np.zeros((*reversed(self.image_shape), len(self.class_to_index)), dtype=np.float32)
        for nom_classe, v in dico_categorie_image.items():
            for bounding_box_corners in v:
                [coin1_x, coin1_y, coin2_x, coin2_y] = bounding_box_corners
                coin1 = np.array([coin1_x, coin1_y]).T
                coin2 = np.array([coin2_x, coin2_y]).T
                ## Calcul des coordonnées après redimensionnement
                matrice_scale_down = np.array([[self.facteur_echelle, 0], [0, self.facteur_echelle]])
                coin1_transfo = np.array(matrice_scale_down.dot(coin1),dtype=np.int)
                coin2_transfo = np.array(matrice_scale_down.dot(coin2),dtype=np.int)
                if abs(coin1_transfo[0] - coin2_transfo[0]) > self.taille_mini_px and abs(
                        coin1_transfo[1] - coin2_transfo[1]) > self.taille_mini_px:
                    label[coin1_transfo[1]:coin2_transfo[1],
                          coin1_transfo[0]:coin2_transfo[0],
                          self.class_to_index[nom_classe]] = 1
        return label

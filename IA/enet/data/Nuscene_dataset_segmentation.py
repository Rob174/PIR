import numpy as np

from IA.model_keras.data.Nuscene_dataset import Nuscene_dataset
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Nuscene_dataset_segmentation(Nuscene_dataset):
    def __init__(self, *args, **kargs):
        super(Nuscene_dataset_segmentation, self).__init__(*args, **kargs)
        self.correspondances_index_classes = {v:k for k,v in Nuscene_dataset.correspondances_classes_index.items()}
        self.get_labels_fct = self.getLabels

    def dataset_stats(self,summary_writer):
        # stats tr
        # for dataset_name,dataset in zip(["tr","valid"],[self.dataset_tr,self.dataset_valid]):
        #     array_compte = np.zeros((len(self.correspondances_index_classes.keys()),))
        #     for i in dataset:
        #         label = self.getLabels(i)
        #         compte_pixel = np.sum(label,axis=[0,1]) # TODO : à détailler comme pour le précédent dataset
        #         array_compte += compte_pixel
        #     fig = plt.figure(figsize=(20, 10))
        #     fig.tight_layout(pad=0)
        #     ax = fig.add_subplot(111)
        #     ax.set_title(f"Statistiques label {self.correspondances_index_classes[]} du dataset {dataset_name}")
        pass # TODO : faire les stats mais par nb de pixels de chaque classe cette fois-ci

    def getLabels(self, index_image,num_batch=None,*args,**kargs):
        """
        Récupération et création des labels d'index index_image dans le fichier json du dataset Nuscene
        :param index_image: index de l'image dans le fichier json des annotations de Nuscene
        :return: np.array de shape (img_width, img_height, #classes) contenant la probabilité que chaque pixel appartienne à chaque classe
        """
        dico_categorie_image = self.content_dataset[index_image]["categories"]
        label = np.zeros((*reversed(self.image_shape), len(self.correspondances_classes_index)), dtype=np.float32)
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
                          self.correspondances_classes_index[nom_classe]] = 1
        return label

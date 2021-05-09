import json
import random

import numpy as np
from PIL import Image
import tensorflow as tf

from IA.model_keras.FolderInfos import FolderInfos
import matplotlib.pyplot as plt
import cv2
import matplotlib

from IA.model_keras.data.Flou_augm import Flou_augment
from IA.model_keras.data.Luminance_augm import Luminance_augment

matplotlib.use('Agg')


class Nuscene_dataset:
    correspondances_classes_index = {
        "animal": 0,
        "human.pedestrian.adult": 1,
        "human.pedestrian.child": 2,
        "human.pedestrian.construction_worker": 3,
        "human.pedestrian.personal_mobility": 4,
        "human.pedestrian.police_officer": 5,
        "human.pedestrian.stroller": 6,
        "human.pedestrian.wheelchair": 7,
        "movable_object.barrier": 8,
        "movable_object.debris": 9,
        "movable_object.pushable_pullable": 10,
        "movable_object.trafficcone": 11,
        "static_object.bicycle_rack": 12,
        "vehicle.bicycle": 13,
        "vehicle.bus.bendy": 14,
        "vehicle.bus.rigid": 15,
        "vehicle.car": 16,
        "vehicle.construction": 17,
        "vehicle.emergency.ambulance": 18,
        "vehicle.emergency.police": 19,
        "vehicle.motorcycle": 20,
        "vehicle.trailer": 21,
        "vehicle.truck": 22
    }
    correspondances_index_classes = None

    def __init__(self, summary_writer, tr_prct: float = 0.6, img_width: int = 1600, limit_nb_tr: int = None, taille_mini_px=10,
                 with_weights="False", batch_size=10,augmentation="f"):
        """

        :param data_folder: chemin vers le dossier data
        :param tr_prct: pourcentage du dataset total consacré à l'entrainement (actualisation des poids)
        :param img_width: taille des images à fournir au réseau
        :param limit_nb_tr: nombre d'image limite pour le dataset d'entrainement et de validation
                            (Nuscene étant très grand, si on veut limiter le nombre d'itérations d'entrainement)
        :param taille_mini_px: taille minimale d'un objet pour qu'il soit considéré comme détectable par le réseau
        :param with_weights: Indique quels poids appliquer aux labels :
                                - False : aucun poids, chaque classe a la même importance lors de l'entrainement
                                - class : donne plus d'importance aux classes sous-représentées :
                                          si la classe chien est sous-représentée, l'erreur pour une image
                                          contenant un chien sera diminuée pour cette classe
                                - classEff : donne plus d'importance aux effectifs de classes sous-représentés :
                                             si il est rare que 100 chiens soit présents sur une image,
                                             si ce cas se présente on va diminuer la loss concernant le nombre de chiens
        :param augmentation: indique si l'on doit réaliser les augmentations
        """
        self.augmentation = augmentation
        if augmentation == "f":
            self.augmentations = []
        elif augmentation == "t":
            self.augmentations = [Luminance_augment,Flou_augment]
        Nuscene_dataset.correspondances_index_classes = {v:k for k,v in Nuscene_dataset.correspondances_classes_index.items()}
        with open("/scratch/rmoine/PIR/extracted_data_nusceneImage.json", 'r') as dataset:
            self.content_dataset = json.load(dataset)
            self.dataset_tr = list(range(0,int(len(self.content_dataset) * tr_prct)))
            self.dataset_valid = list(range(int(len(self.content_dataset) * tr_prct),len(self.content_dataset)))
            self.batch_size = batch_size
            # Récupère la taille des images
            self.root_dir = "/scratch/rmoine/PIR/nuscene/"
            width, height = Image.open(self.root_dir + self.content_dataset[0]["imageName"]).size  # 1600x900
            self.facteur_echelle = img_width / width
            self.image_shape = (img_width, int(img_width / width * height))
            self.taille_mini_px = taille_mini_px
            if limit_nb_tr is not None:
                self.limit_nb_tr = limit_nb_tr
            else:
                self.limit_nb_tr = len(self.dataset_tr)
            # statistiques du dataset
            self.dataset_stats(summary_writer)
            self.get_labels_fct = None
            if with_weights == "False":
                self.get_labels_fct = self.getLabelsWithUnitWeight
            elif with_weights == "class":
                self.get_labels_fct = self.getLabelsWithWeightsPerClass
            elif with_weights == "classEff":
                self.get_labels_fct = self.getLabelsWithWeightsPerClassEff
            else:
                raise Exception("Class weight argument not recognized")
    def dataset_stats(self,summary_writer):

        self.stat_per_class_eff_tr = {classe: {} for classe in self.correspondances_classes_index.keys()}
        self.stat_per_class_tr = {classe: 0 for classe in self.correspondances_classes_index.keys()}
        for index_img in self.dataset_tr:
            label = self.getLabels(index_img)
            for index_class in range(len(label)):
                self.stat_per_class_tr[self.correspondances_index_classes[index_class]] += label[index_class]
                effectif = str(int(label[index_class]))
                if effectif not in self.stat_per_class_eff_tr[self.correspondances_index_classes[index_class]].keys():
                    self.stat_per_class_eff_tr[self.correspondances_index_classes[index_class]][effectif] = 0
                self.stat_per_class_eff_tr[self.correspondances_index_classes[index_class]][effectif] += 1

        self.stat_per_class_eff_valid = {classe: {} for classe in self.correspondances_classes_index.keys()}
        self.stat_per_class_valid = {classe: 0 for classe in self.correspondances_classes_index.keys()}
        for index_img in self.dataset_valid:
            label = self.getLabels(index_img)
            for index_class in range(len(label)):
                self.stat_per_class_valid[self.correspondances_index_classes[index_class]] += label[index_class]
                effectif = str(int(label[index_class]))
                if effectif not in self.stat_per_class_eff_valid[
                    self.correspondances_index_classes[index_class]].keys():
                    self.stat_per_class_eff_valid[self.correspondances_index_classes[index_class]][effectif] = 0
                self.stat_per_class_eff_valid[self.correspondances_index_classes[index_class]][effectif] += 1

        # Ploting distributions
        with summary_writer.as_default():
            for dico, name in zip([self.stat_per_class_eff_tr, self.stat_per_class_eff_valid],
                                  ["training dataset", "validation dataset"]):
                for classe, dico_classe in dico.items():
                    range_min = min(map(int, dico_classe.keys()))
                    range_max = max(map(int, dico_classe.keys()))
                    vecteur_x = np.arange(range_min, range_max + 1)
                    vecteur_bar = np.zeros((range_max - range_min + 1,))
                    for effectif, nb_fois_vu in dico_classe.items():
                        vecteur_bar[int(effectif) - range_min] = nb_fois_vu
                    plt.clf()
                    fig = plt.figure(figsize=(20, 10))
                    fig.tight_layout(pad=0)
                    ax = fig.add_subplot(111)
                    ax.set_title(f"Statistiques label {classe} du {name}")
                    ax.bar(vecteur_x, vecteur_bar, log=True)
                    ax.set_xticks(vecteur_x)
                    ax.set_xlabel("Nombre d'objets présents dans 1 image")
                    ax.set_ylabel("Nombre de fois où cette situation se produit")
                    # get image in numpy array (thanks to https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array)
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    data = np.stack((data,), axis=0)

                    tf.summary.image(f"dataset_{name.split()[0]}_{classe}_stats", data, step=0)
                    summary_writer.flush()
                    plt.close()
        print("--------------------------------------STAT DONE--------------------------------------------------")
    def getImage(self, index_image):
        """
        Récupération de l'image de path précisé à l'index index_image du fichier json représentant le dataset Nuscene
        :param index_image: index du dictionnaire donnant le chemin de l'image index_image dans le fichier json représentant le dataset Nuscene
        :return: image normaliséee np.array de shape (#image_shape (calculé à partir de img_width du constructeur), #image_shape, 3)
        """
        path = self.root_dir + self.content_dataset[index_image]["imageName"]
        image = Image.open(path)
        image = image.resize(self.image_shape)
        image = np.array(image,dtype=np.float32) / 255.
        if self.augmentation == 't':
            for augmenteur in self.augmentations:
                image = augmenteur.augment(image)
        return image

    def getLabels(self, index_image):
        """
        Récupération et création des labels d'index index_image dans le fichier json du dataset Nuscene
        :param index_image: index de l'image dans le fichier json des annotations de Nuscene
        :return: np.array de shape (#nb_classes, ) contenant l'effectif d'apparition de chaque classe sur cette image
        """
        dico_categorie_image = self.content_dataset[index_image]["categories"]
        label = np.zeros((len(Nuscene_dataset.correspondances_classes_index.values())))
        for k, v in dico_categorie_image.items():
            for bounding_box_corners in v:
                [coin1_x, coin1_y, coin2_x, coin2_y] = bounding_box_corners
                coin1 = np.array([coin1_x, coin1_y]).T
                coin2 = np.array([coin2_x, coin2_y]).T
                ## Calcul des coordonnées après redimensionnement
                matrice_scale_down = np.array([[self.facteur_echelle, 0], [0, self.facteur_echelle]])
                coin1_transfo = matrice_scale_down.dot(coin1)
                coin2_transfo = matrice_scale_down.dot(coin2)
                if abs(coin1_transfo[0] - coin2_transfo[0]) > self.taille_mini_px and abs(
                        coin1_transfo[1] - coin2_transfo[1]) > self.taille_mini_px:
                    label[self.correspondances_classes_index[k]] += 1
        return label

    def getLabelsWithUnitWeight(self, index, dataset="tr"):
        """
        Génère les labels (nombre d'objets de chaque classe présent sur une image et les poids associés
        Ici on ne pondère pas chaque classe
        :param index_image: index de l'image dans le fichier json du dataset Nuscene
        :return: np.array, shape (2,#classes)
        """
        label = self.getLabels(index)
        return np.stack((label, np.ones(label.shape)), axis=0)

    def getLabelsWithWeightsPerClass(self, index, dataset="tr"):
        """
        Génère les labels (nombre d'objets de chaque classe présent sur une image et les poids associés
        Ici on pondère chaque classe par son pourcentage d'apparition dans le dataset pour donner plus de poids aux classes sous-représentées
        :param index_image: index de l'image dans le fichier json du dataset Nuscene
        :return: np.array, shape (2,#classes)
        """
        label = self.getLabels(index)
        poids = np.zeros(label.shape)
        if dataset == "tr":
            dico_stat = self.stat_per_class_tr
        elif dataset == "valid":
            dico_stat = self.stat_per_class_valid
        else:
            raise Exception(f"getLabelsWithWeightsPerClass : dataset unrecognized : {dataset}")
        for i in range(len(label)):
            nom_classe = Nuscene_dataset.correspondances_index_classes[i]
            effectif = label[i]
            poids[i] = dico_stat[nom_classe] if effectif > 0 else 0
        total = float(sum(v for v in dico_stat.values()))
        poids /= total
        return np.stack((label, poids), axis=0)

    def getLabelsWithWeightsPerClassEff(self, index, dataset="tr"):
        """
        Génère les labels (nombre d'objets de chaque classe présent sur une image et les poids associés
        Ici on pondère chaque classe par la fréquence de prédiction de cet effectif pour chaque classe :
        Exemple : si le fait de trouver 10 voitures sur une image
                  apparait 3 fois sur les x images du dataset d'entrainement
                  alors le poids sera de 3/x pour la classe voiture
        :param index_image: index de l'image dans le fichier json du dataset Nuscene
        :return: np.array, shape (2,#classes)
        """
        label = self.getLabels(index)
        poids = np.zeros(label.shape)
        if dataset == "tr":
            dico_stat = self.stat_per_class_eff_tr
        elif dataset == "valid":
            dico_stat = self.stat_per_class_eff_valid
        else:
            raise Exception(f"getLabelsWithWeightsPerClass : dataset unrecognized : {dataset}")
        for i in range(len(label)):
            nom_classe = Nuscene_dataset.correspondances_index_classes[i]
            effectif = label[i]
            try:
                poids[i] = dico_stat[nom_classe][str(int(effectif))]
            except:
                raise Exception(f"Key {str(int(effectif))} not found in class {nom_classe} with {self.stat_per_class_eff[nom_classe]}")
        total = len(self.dataset_tr)
        poids /= total
        return np.stack((label, poids), axis=0)

    def getNextBatchTr(self):
        """
        Générateur construisant le prochain batch d'entrainement
        :return: images labels et poids (pour chaque classe) : Tuple(
                                                                array de shape (#nb_img_par_batch, #taille_img_x, #taille_img_y, #canaux),
                                                                array de shape (#nb_img_par_batch, 2, #nb_classes)
                                                                )
        """
        bufferLabel, bufferImg = [], []
        random.shuffle(self.dataset_tr)
        for i in self.dataset_tr[:self.limit_nb_tr]:
            bufferImg.append(self.getImage(i))
            bufferLabel.append(self.get_labels_fct(i,dataset="tr"))
            if len(bufferImg) % self.batch_size == 0 and i > 0:
                batches = np.stack(bufferImg, axis=0), np.stack(bufferLabel, axis=0)
                bufferLabel, bufferImg = [], []
                yield batches

    def getNextBatchValid(self):
        """
        Générateur construisant le prochain batch de validation
        :return: images labels et poids (pour chaque classe) : Tuple(
                                                                array de shape (#nb_img_par_batch, #taille_img_x, #taille_img_y, #canaux),
                                                                array de shape (#nb_img_par_batch, 2, #nb_classes)
                                                                )
        """
        bufferLabel, bufferImg = [], []
        random.shuffle(self.dataset_valid)
        while True:
            for i in self.dataset_valid:
                bufferImg.append(self.getImage(i))
                bufferLabel.append(self.get_labels_fct(i,dataset="valid"))
                if len(bufferImg) % self.batch_size == 0 and i > 0:
                    batches = np.stack(bufferImg, axis=0), np.stack(bufferLabel, axis=0)
                    bufferLabel, bufferImg = [], []
                    yield batches

    def getNextBatchFullDataset(self):
        bufferLabel, bufferImg = [], []
        full_dataset = list(range(int(len(self.content_dataset))))
        for i in full_dataset:
            bufferImg.append(self.getImage(i))
            bufferLabel.append(self.get_labels_fct(i, dataset="valid" if i in self.dataset_valid else "tr"))
            if len(bufferImg) % self.batch_size == 0 and i > 0:
                batches = np.stack(bufferImg, axis=0), np.stack(bufferLabel, axis=0)
                bufferLabel, bufferImg = [], []
                yield batches
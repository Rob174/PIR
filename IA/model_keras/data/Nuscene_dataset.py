import json
import random

import numpy as np
from PIL import Image

from model_keras.FolderInfos import FolderInfos


class Nuscene_dataset:
    correspondances_classes = {
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

    def __init__(self, data_folder: str, tr_prct: float = 0.6, img_width: int = 1600, limit_nb_tr: int = None, taille_mini_px=10,
                 with_weights="False", batch_size=10):
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
        """
        with open("/scratch/rmoine/PIR/extracted_data_nusceneImage.json", 'r') as dataset:
            self.content_dataset = json.load(dataset)
            self.dataset_tr = self.content_dataset[:int(len(self.content_dataset) * tr_prct)]
            self.dataset_valid = self.content_dataset[int(len(self.content_dataset) * tr_prct):]
            self.batch_size = batch_size
            # Récupère la taille des images
            self.root_dir = "/scratch/rmoine/PIR/nuscene/"
            width, height = Image.open(self.root_dir + self.content_dataset[0]["imageName"]).size  # 1600x900
            self.facteur_echelle = img_width / width
            self.image_shape = (img_width, int(img_width / width * height))
            self.taille_mini_px = taille_mini_px
            print("shape : ", self.image_shape)
            if limit_nb_tr is not None:
                self.limit_nb_tr = limit_nb_tr
            else:
                self.limit_nb_tr = len(self.dataset_tr)
            path_stat_per_class_eff = data_folder + "/2021-04-19_12h06min43s_class_distribution_nuscene/2021-04-19_12h06min43s_class_distribution_nuscenestatistics.json"
            path_stat_per_class = data_folder + "/2021-04-19_12h06min43s_class_distribution_nuscene/2021-04-19_12h06min43s_class_distribution_stat_nb_elem_per_class.json"
            with open(path_stat_per_class, "r") as fp:
                self.stat_per_class = json.load(fp)
            with open(path_stat_per_class_eff, "r") as fp:
                self.stat_per_class_eff = json.load(fp)

            self.get_labels_fct = None
            if with_weights == "False":
                self.get_labels_fct = self.getLabelsWithUnitWeight
            elif with_weights == "class":
                self.get_labels_fct = self.getLabelsWithWeightsPerClass
            elif with_weights == "classEff":
                self.get_labels_fct = self.getLabelsWithWeightsPerClassEff
            else:
                raise Exception("Class weight argument not recognized")

    def getImage(self, index_image):
        """
        Récupération de l'image de path précisé à l'index index_image du fichier json représentant le dataset Nuscene
        :param index_image: index du dictionnaire donnant le chemin de l'image index_image dans le fichier json représentant le dataset Nuscene
        :return: image normaliséee np.array de shape (#image_shape (calculé à partir de img_width du constructeur), #image_shape, 3)
        """
        path = self.root_dir + self.content_dataset[index_image]["imageName"]
        image = Image.open(path)
        image = image.resize(self.image_shape)
        image = np.array(image) / 255.
        return image

    def getLabels(self, index_image):
        """
        Récupération et création des labels d'index index_image dans le fichier json du dataset Nuscene
        :param index_image: index de l'image dans le fichier json des annotations de Nuscene
        :return: np.array de shape (#nb_classes, ) contenant l'effectif d'apparition de chaque classe sur cette image
        """
        dico_categorie_image = self.content_dataset[index_image]["categories"]
        label = np.zeros((len(Nuscene_dataset.correspondances_classes.values())))
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
                    label[self.correspondances_classes[k]] += 1
        return label

    def getLabelsWithUnitWeight(self, index_image):
        """
        Génère les labels (nombre d'objets de chaque classe présent sur une image et les poids associés
        Ici on ne pondère pas chaque classe
        :param index_image: index de l'image dans le fichier json du dataset Nuscene
        :return: np.array, shape (2,#classes)
        """
        label = self.getLabels(index_image)
        return np.stack((label, np.ones(label.shape)), axis=0)

    def getLabelsWithWeightsPerClass(self, index_image):
        """
        Génère les labels (nombre d'objets de chaque classe présent sur une image et les poids associés
        Ici on pondère chaque classe par son pourcentage d'apparition dans le dataset pour donner plus de poids aux classes sous-représentées
        :param index_image: index de l'image dans le fichier json du dataset Nuscene
        :return: np.array, shape (2,#classes)
        """
        label = self.getLabels(index_image)
        poids = np.zeros(label.shape)
        for i in range(len(label)):
            nom_classe = [k for k, v in self.correspondances_classes.items() if v == i][0]
            effectif = label[i]
            poids[i] = self.stat_per_class[nom_classe] if effectif > 0 else 0
        total = float(sum(v for v in self.stat_per_class.values()))
        poids /= total
        return np.stack((label, poids), axis=0)

    def getLabelsWithWeightsPerClassEff(self, index_image):
        """
        Génère les labels (nombre d'objets de chaque classe présent sur une image et les poids associés
        Ici on pondère chaque classe par la fréquence de prédiction de cet effectif pour chaque classe :
        Exemple : si le fait de trouver 10 voitures sur une image
                  apparait 3 fois sur les x images du dataset d'entrainement
                  alors le poids sera de 3/x pour la classe voiture
        :param index_image: index de l'image dans le fichier json du dataset Nuscene
        :return: np.array, shape (2,#classes)
        """
        label = self.getLabels(index_image)
        poids = np.zeros(label.shape)
        for i in range(len(label)):
            nom_classe = [k for k, v in self.correspondances_classes.items() if v == i][0]
            effectif = label[i]
            print(nom_classe,":",self.stat_per_class_eff[nom_classe])
            poids[i] = self.stat_per_class_eff[nom_classe][str(int(effectif))]
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
        index_imgs = list(range(len(self.dataset_tr)))
        random.shuffle(index_imgs)
        for i in range(min(self.limit_nb_tr, len(self.dataset_tr))):
            bufferImg.append(self.getImage(i))
            bufferLabel.append(self.get_labels_fct(i))
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
        index_imgs = list(range(len(self.dataset_valid)))
        random.shuffle(index_imgs)
        while True:
            for i in range(len(self.dataset_valid)):
                bufferImg.append(self.getImage(i))
                bufferLabel.append(self.get_labels_fct(i))
                if len(bufferImg) % self.batch_size == 0 and i > 0:
                    batches = np.stack(bufferImg, axis=0), np.stack(bufferLabel, axis=0)
                    bufferLabel, bufferImg = [], []
                    yield batches

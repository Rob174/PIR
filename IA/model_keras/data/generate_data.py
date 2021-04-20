import json
import numpy as np
from PIL import Image
import random

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
    def __init__(self,tr_prct: float =0.6,img_width: int =1600,limit_nb_tr: int =None,taille_mini_px=10):
        with open("/scratch/rmoine/PIR/extracted_data_nusceneImage.json", 'r') as dataset:
            self.content_dataset = json.load(dataset)
            self.dataset_tr = self.content_dataset[:int(len(self.content_dataset)*tr_prct)]
            self.dataset_valid = self.content_dataset[int(len(self.content_dataset)*tr_prct):]
            self.batch_size=10
            # Récupère la taille des images
            self.root_dir= "/scratch/rmoine/PIR/nuscene/"
            width, height = Image.open(self.root_dir + self.content_dataset[0]["imageName"]).size # 1600x900
            self.facteur_echelle = img_width/width
            self.image_shape = (img_width,int(img_width/width*height))
            self.taille_mini_px = taille_mini_px
            print("shape : ",self.image_shape)
            if limit_nb_tr is not None:
                self.limit_nb_tr = limit_nb_tr
            else:
                self.limit_nb_tr = len(self.dataset_tr)

    def getImage(self, index_image):
        path = self.root_dir + self.content_dataset[index_image]["imageName"]
        image = Image.open(path)
        image = image.resize(self.image_shape)
        image = np.array(image) / 255.
        return image

    def getLabels(self, index_image):
        dico_categorie_image = self.content_dataset[index_image]["categories"]
        label = np.zeros((len(Nuscene_dataset.correspondances_classes.values())))
        for k, v in dico_categorie_image.items():
            for bounding_box_corners in v:
                [coin1_x,coin1_y,coin2_x,coin2_y] = bounding_box_corners
                coin1 = np.array([coin1_x,coin1_y]).T
                coin2 = np.array([coin2_x,coin2_y]).T
                ## Calcul des coordonnées après redimensionnement
                matrice_scale_down = np.array([[self.facteur_echelle,0],[0,self.facteur_echelle]])
                coin1_transfo = matrice_scale_down.dot(coin1)
                coin2_transfo = matrice_scale_down.dot(coin2)
                if abs(coin1_transfo[0] - coin2_transfo[0]) > self.taille_mini_px and abs(coin1_transfo[1] - coin2_transfo[1]) > self.taille_mini_px:
                    label[self.correspondances_classes[k]] += 1
        return label
    def getNextBatchTr(self):
        bufferLabel, bufferImg = [], []
        index_imgs = list(range(len(self.dataset_tr)))
        random.shuffle(index_imgs)
        for i in range(min(self.limit_nb_tr,len(self.dataset_tr))):
            bufferImg.append(self.getImage(i))
            bufferLabel.append(self.getLabels(i))
            if len(bufferImg) % self.batch_size == 0 and i > 0:
                batches = np.stack(bufferImg, axis=0), np.stack(bufferLabel, axis=0)
                bufferLabel, bufferImg = [], []
                yield batches
    def getNextBatchValid(self):
        bufferLabel, bufferImg = [], []
        index_imgs = list(range(len(self.dataset_valid)))
        random.shuffle(index_imgs)
        while True:
            for i in range(len(self.dataset_valid)):
                bufferImg.append(self.getImage(i))
                bufferLabel.append(self.getLabels(i))
                if len(bufferImg) % self.batch_size == 0 and i > 0:
                    batches = np.stack(bufferImg, axis=0), np.stack(bufferLabel, axis=0)
                    bufferLabel, bufferImg = [], []
                    yield batches
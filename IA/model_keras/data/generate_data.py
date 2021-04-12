import json
import numpy as np
from PIL import Image
import random

class Nexet_dataset:
    def __init__(self):
        with open("/scratch/rmoine/PIR/extracted_data_nusceneImage.json", 'r') as dataset:
            self.content_dataset = json.load(dataset)
            self.correspondances_classes = {
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
                "static_object.bicycle_rack *": 12,
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
            self.nb_images= len(self.content_dataset)
            self.batch_size=10
            self.root_dir= "/scratch/rmoine/PIR/nuscene/"
    def getImage(self, index_image):
        path = self.root_dir + self.content_dataset[index_image]["imageName"]
        image = np.array(Image.open(path))
        return image

    def getLabels(self, index_image):
        dico_categorie_image = self.content_dataset[index_image]["categories"]
        nb_boundingbox = 0
        label = np.zeros((len(self.correspondances_classes.values())))
        for k, v in dico_categorie_image.items():
            nb_boundingbox += len(v)
            label[self.correspondances_classes[k]] += len(v)
        label = label / nb_boundingbox
        return label
    def getNextBatch(self):
        bufferLabel, bufferImg = [], []
        index_imgs = list(range(len(self.content_dataset)))
        random.shuffle(index_imgs)
        for i in range(len(self.content_dataset)):
            bufferImg.append(self.getImage(i))
            bufferLabel.append(self.getLabels(i))
            if len(bufferImg) % self.batch_size == 0 and i > 0:
                batches = np.stack(bufferImg, axis=0), np.stack(bufferLabel, axis=0)
                bufferLabel, bufferImg = [], []
                yield batches

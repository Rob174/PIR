from typing import List

import scipy.misc
import random
from PIL import Image
import numpy as np

class ImageSteeringDB(object):
    """Preprocess images of the road ahead ans steering angles."""

    def __init__(self,batch_size):
        imgs_path: List[str] = []
        angle_values: List[float] = []

        # read data.txt
        data_path ="/scratch/rmoine/PIR/PilotNet/data/datasets/driving_dataset/"

        with open(data_path + "data.txt") as f:
            for line in f:
                path = data_path + line.split()[0]
                imgs_path.append(path[-150:])
                # the paper by Nvidia uses the inverse of the turning radius,
                # but steering wheel angle is proportional to the inverse of turning radius
                # so the steering wheel angle in radians is used as the output
                angle_values.append(float(line.split()[1]) * scipy.pi / 180)

        # get number of images
        self.num_images = len(imgs_path)
        self.batch_size = batch_size

        self.train_imgs_path: List[str] = imgs_path[:int(self.num_images * 0.8)]
        self.train_angles_values: List[float] = angle_values[:int(self.num_images * 0.8)]

        self.val_imgs_path: List[str] = imgs_path[-int(self.num_images * 0.2):]
        self.val_angles_values: List[float] = angle_values[-int(self.num_images * 0.2):]

        self.num_train_images = len(self.train_imgs_path)
        self.num_val_images = len(self.val_imgs_path)

    def train_batch_generator(self):
        batch_imgs = []
        batch_angles = []
        index_elems: List[int] = list(range(self.num_train_images))
        # Mélange les éléments (on les récupèrera par leur index)
        random.shuffle(index_elems)
        for i,index_elem in enumerate(index_elems):
            img_path: str = self.train_imgs_path[index_elem]
            angle_value: float = self.train_angles_values[index_elem]
            img = Image.open(img_path).resize(size=(200, 66))
            img = np.array(img) / 255.0
            batch_imgs.append(img)
            batch_angles.append(angle_value)
            if len(batch_imgs) % self.batch_size == 0 and i > 0:
                batches = np.stack(batch_imgs, axis=0), np.stack(batch_angles, axis=0)
                batch_imgs, batch_angles = [], []
                yield batches

    def val_batch_generator(self):
        batch_imgs = []
        batch_angles = []
        index_elems: List[int] = list(range(self.num_val_images))
        # Mélange les éléments (on les récupèrera par leur index)
        random.shuffle(index_elems)
        for i,index_elem in enumerate(index_elems):
            img_path: str = self.val_imgs_path[index_elem]
            angle_value: float = self.val_angles_values[index_elem]
            img = Image.open(img_path).resize(size=(200, 66))
            img = np.array(img)
            batch_imgs.append(img / 255.0)
            batch_angles.append(angle_value)
            if len(batch_imgs) % self.batch_size == 0 and i > 0:
                batches = np.stack(batch_imgs, axis=0), np.stack(batch_angles, axis=0)
                batch_imgs, batch_angles = [], []
                yield batches

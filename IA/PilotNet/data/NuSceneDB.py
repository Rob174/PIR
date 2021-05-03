from typing import List
import os.path

import scipy.misc
import random
from PIL import Image
import numpy as np
import tensorflow as tf

# from nuscenes_devkit.python_sdk.nuscenes import nuscenes
import json


class NuSceneDB(object):
    # Every label present in the nuScene dataset
    labels = ['animal',
            'human.pedestrian.adult',
            'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.personal_mobility',
            'human.pedestrian.police_officer',
            'human.pedestrian.stroller',
            'human.pedestrian.wheelchair',
            'movable_object.barrier',
            'movable_object.debris',
            'movable_object.pushable_pullable',
            'movable_object.trafficcone',
            'static_object.bicycle_rack',
            'vehicle.bicycle',
            'vehicle.bus.bendy',
            'vehicle.bus.rigid',
            'vehicle.car',
            'vehicle.construction',
            'vehicle.emergency.ambulance',
            'vehicle.emergency.police',
            'vehicle.motorcycle',
            'vehicle.trailer',
            'vehicle.truck']

    """Preprocess images of the road ahead and expected outputs."""

    def __init__(self, batch_size, tr_prct=0.6):
        imgs: List[str] = []  # Will store the path of each image
        outputs: List[float] = []  # Will store the occurrence of each label for each image

        # Extract images annotations
        data_path = "/scratch/rmoine/PIR/nuscene/"
        json_path = "/scratch/rmoine/PIR/extracted_data_nusceneImage.json"

        with open(json_path) as json_file:
            images = json.loads(json_file.read())

            for image in images:
                filename = image['imageName']
                categories = image['categories']

                image_path = data_path + filename

                if os.path.isfile(image_path):
                    image_outputs = [0] * len(NuSceneDB.labels)

                    for detectedCategory in list(categories):
                        # Retrieve category array of bounding boxes
                        bounding_boxes = categories[detectedCategory]

                        # Count the number of occurrences for the detected category
                        occurrences = len(bounding_boxes)

                        # Retrieve index of the label within 'labels' array
                        label_index = NuSceneDB.labels.index(detectedCategory)

                        # Store the number of occurrences within the right index of the 'imageOutputs' array
                        image_outputs[label_index] = occurrences

                    imgs.append(data_path + filename)
                    outputs.append(image_outputs)

        self.dataset_size = len(imgs)
        self.training_dataset_size = self.dataset_size * tr_prct
        self.validation_dataset_size = self.dataset_size - self.training_dataset_size

        print("[NuSceneDB] number of images within the entire dataset : " + str(self.dataset_size))
        print("[NuSceneDB] number of images within the training dataset : " + str(self.training_dataset_size))
        print("[NuSceneDB] number of images within the validation dataset : " + str(self.validation_dataset_size))

        self.dataset_tr = list(range(0, int(len(imgs) * tr_prct)))
        self.dataset_valid = list(range(int(len(imgs) * tr_prct), len(imgs)))

        # get number of images
        self.num_images = len(imgs)
        self.batch_size = batch_size

        self.imgs_path = imgs
        self.outputs_values = outputs

        self.train_imgs_path: List[str] = imgs[:int(self.num_images * 0.8)]
        self.train_outputs_values: List[float] = outputs[:int(self.num_images * 0.8)]

        self.val_imgs_path: List[str] = imgs[-int(self.num_images * 0.2):]
        self.val_outputs_values: List[float] = outputs[-int(self.num_images * 0.2):]

        self.num_train_images = len(self.train_imgs_path)
        self.num_val_images = len(self.val_imgs_path)

    def train_batch_generator(self):
        batch_imgs = []
        batch_outputs = []

        # Shuffle the elements, will be retrieved by their index
        random.shuffle(self.dataset_tr)

        for index_elem in self.dataset_tr:
            img_path: str = self.imgs_path[index_elem]
            outputs_value: float = self.outputs_values[index_elem]
            img = Image.open(img_path).resize(size=(200, 66))
            img = np.array(img) / 255.0
            batch_imgs.append(img)
            batch_outputs.append(outputs_value)
            if len(batch_imgs) % self.batch_size == 0 and len(batch_imgs) > 0:
                batches = np.stack(batch_imgs, axis=0), np.stack(batch_outputs, axis=0)
                batch_imgs, batch_outputs = [], []
                yield batches

    def val_batch_generator(self):
        batch_imgs = []
        batch_outputs = []

        # Shuffle the elements, will be retrieved by their index
        random.shuffle(self.dataset_valid)

        for index_elem in self.dataset_valid:
            img_path: str = self.imgs_path[index_elem]
            outputs_value: float = self.outputs_values[index_elem]
            img = Image.open(img_path).resize(size=(200, 66))
            img = np.array(img)
            batch_imgs.append(img / 255.0)
            batch_outputs.append(outputs_value)
            if len(batch_imgs) % self.batch_size == 0 and len(batch_imgs) > 0:
                batches = np.stack(batch_imgs, axis=0), np.stack(batch_outputs, axis=0)
                batch_imgs, batch_outputs = [], []
                yield batches

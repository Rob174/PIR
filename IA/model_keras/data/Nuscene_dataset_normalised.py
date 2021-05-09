from IA.model_keras.data.Nuscene_dataset import Nuscene_dataset
import numpy as np


class Nuscene_dataset_normalized(Nuscene_dataset):
    def __init__(self,*args,**kargs):
        super(Nuscene_dataset_normalized, self).__init__(*args,**kargs)
    def getLabels(self, index_image):
        labels = super(Nuscene_dataset_normalized, self).getLabels(index_image)
        return labels / np.sum(labels)
    def dataset_stats(self,summary_writer):
        pass
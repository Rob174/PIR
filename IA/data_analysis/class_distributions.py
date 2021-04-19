import os
import sys


chemin_fichier = os.path.realpath(__file__).split("/")
sys.path.append("/".join(chemin_fichier[:-2]))
sys.path.append("/".join(chemin_fichier[:-3]))


from IA.model_keras.data.generate_data import Nuscene_dataset
import json
from IA.model_keras.FolderInfos import FolderInfos

dataset = Nuscene_dataset()

iteratorTr = dataset.getNextBatchTr()
iteratorValid = dataset.getNextBatchValid()


def add_stat(batch_label):
    global dico_eff
    for batch_index in range(batch_label.shape[0]):
        for classe_index in range(batch_label.shape[1]):
            key = str(batch_label[batch_index, classe_index])
            if key not in dico_eff[str(classe_index)].keys():
                dico_eff[str(classe_index)][key] = 0
            dico_eff[str(classe_index)][key] += 1


batchImg, batchLabel = next(iteratorTr)
dico_eff = {str(i): {} for i in range(batchLabel.shape[1])}
add_stat(batchLabel)

while True:
    try:
        batchImg, batchLabel = next(iteratorTr)
        add_stat(batchLabel)
    except StopIteration:
        print("Tr dataset ok")
        break

while True:
    try:
        batchImg, batchLabel = next(iteratorValid)
        add_stat(batchLabel)
    except StopIteration:
        print("Valid dataset ok")
        break

FolderInfos.init(custom_name="class_distribution_nuscene")
with open(FolderInfos.base_filename+"statistics.json","w") as fp:
    json.dump(dico_eff,fp)
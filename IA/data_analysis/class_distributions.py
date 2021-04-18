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
            dico_eff[str(classe_index)][str(batch_label[batch_index, classe_index])] += 1


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
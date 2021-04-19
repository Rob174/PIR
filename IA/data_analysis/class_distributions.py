import os
import sys

chemin_fichier = os.path.realpath(__file__).split("/")
sys.path.append("/".join(chemin_fichier[:-2]))
sys.path.append("/".join(chemin_fichier[:-3]))

from IA.model_keras.data.generate_data import Nuscene_dataset
import json
from IA.model_keras.FolderInfos import FolderInfos

with open("/scratch/rmoine/PIR/extracted_data_nusceneImage.json", "r") as fp:
    file = json.load(fp)

dico_eff = {k: {} for k in Nuscene_dataset.correspondances_classes.keys()}

for img_dico in file:
    dico_categorie_image = img_dico["categories"]
    nb_boundingbox = 0
    for classe, v in dico_categorie_image.items():
        nb_obj = str(len(v))
        if nb_obj not in dico_eff[classe].keys():
            dico_eff[classe][nb_obj] = 0
        dico_eff[classe][nb_obj] += 1

FolderInfos.init(custom_name="class_distribution_nuscene")
with open(FolderInfos.base_filename + "statistics.json", "w") as fp:
    json.dump(dico_eff, fp,indent=4)
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

liste_classes = Nuscene_dataset.correspondances_classes_index.keys()
dico_eff = {k: {} for k in liste_classes}

for img_dico in file:
    dico_categorie_image = img_dico["categories"]
    nb_boundingbox = 0
    for classe, v in dico_categorie_image.items():
        nb_obj = str(len(v))
        if nb_obj not in dico_eff[classe].keys():
            dico_eff[classe][nb_obj] = 0
        dico_eff[classe][nb_obj] += 1
    for classe in [k for k in liste_classes if k not in dico_categorie_image.keys()]:
        if "0" not in dico_eff[classe].keys():
            dico_eff[classe]["0"] = 0
        dico_eff[classe]["0"] += 1

FolderInfos.init(custom_name="class_distribution_nuscene")
with open(FolderInfos.base_filename + "statistics.json", "w") as fp:
    json.dump(dico_eff, fp,indent=4)

# Calcul des statistiques par classe
dico_par_classe = {k:0 for k in dico_eff.keys()}
for kclass,dico_ss_eff in dico_eff.items():
    for keff,value in dico_ss_eff.items():
        if keff != "0":
            dico_par_classe[kclass] += value
with open(FolderInfos.base_filename + "statistics_per_class.json", "w") as fp:
    json.dump(dico_par_classe, fp,indent=4)

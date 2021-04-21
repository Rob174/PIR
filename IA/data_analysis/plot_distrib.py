import json
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    with open("C:/Users/robin/Documents/projets/PIR/data/" +
              "2021-04-21_17h21min29s_class_distribution_nuscene/" +
              "2021-04-21_17h21min29s_class_distribution_nuscenestatistics.json") as fp:
        dico = json.load(fp)

    folder_output = "C:/Users/robin/Documents/projets/PIR/data/2021-04-21_17h21min29s_class_distribution_nuscene/2021-04-21_17h21min29s"

    for classe,dico_classe in dico.items():
        range_min = min(map(int,dico_classe.keys()))
        range_max = max(map(int,dico_classe.keys()))
        vecteur_x = np.arange(range_min,range_max+1)
        vecteur_bar = np.zeros((range_max-range_min+1,))
        for effectif,nb_fois_vu in dico_classe.items():
            vecteur_bar[int(effectif)-range_min] = nb_fois_vu
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.title(f"Statistiques label {classe}")
        plt.bar(vecteur_x,vecteur_bar,log=True)
        plt.xticks(vecteur_x)
        plt.xlabel("Nombre d'objets présents dans 1 image")
        plt.ylabel("Nombre de fois où cette situation se produit")
        plt.savefig(folder_output+classe+".png")
        plt.close()
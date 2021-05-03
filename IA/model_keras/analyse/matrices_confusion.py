from typing import Dict, List
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from numpy.core.defchararray import add as addStr

from IA.model_keras.FolderInfos import FolderInfos

class MakeConfusionMatrix:
    def __init__(self,model,dataset,nb_classes,correspondances_index_classes,summary_writer):
        self.model = model
        self.dataset = dataset
        self.nb_classes = nb_classes
        self.correspondances_index_classes = correspondances_index_classes
        self.summary_writer = summary_writer
    def add_sample(self,batch_img, batch_true):
        batch_pred = self.model.predict(batch_img)
        batch_true = np.array(np.round(batch_true[:, 0, :].numpy()), dtype=np.int)
        batch_pred = np.array(np.round(batch_pred[:,0,:]),dtype=np.int)
        for i_batch in range(len(batch_pred)):
            for i_classe in range(self.nb_classes):
                self.matrices_confusion[i_classe]["true"].append(batch_true[i_batch, i_classe])
                self.matrices_confusion[i_classe]["pred"].append(batch_pred[i_batch, i_classe])
    def save_confusion_matrix(self,matrice_confusion_petite,labels,classe_index,identifieur_nom=""):
        total = np.sum(matrice_confusion_petite)
        nb_val = len(labels)
        matrice_confusion_prct_petite = matrice_confusion_petite / total * 100
        matrice_confusion_prct = np.zeros((nb_val + 1, nb_val + 1))
        matrice_confusion_prct[:nb_val, :nb_val] = matrice_confusion_prct_petite
        matrice_confusion_prct[nb_val, nb_val] = np.sum(np.diag(matrice_confusion_prct_petite))

        matrice_confusion = np.zeros((nb_val + 1, nb_val + 1))
        matrice_confusion[:nb_val, :nb_val] = matrice_confusion_petite
        matrice_confusion[nb_val, nb_val] = np.sum(np.diag(matrice_confusion_petite))
        top3voisins_precision = np.sum(np.diag(matrice_confusion_prct_petite)) \
                                + np.sum(np.diag(matrice_confusion_prct_petite, k=-1)) \
                                + np.sum(np.diag(matrice_confusion_prct_petite, k=1))
        matrice_confusion_prct = np.round(matrice_confusion_prct, decimals=2)
        matrice_confusion_prct_petite = np.round(matrice_confusion_prct_petite, decimals=2)
        top3voisins_precision = np.round(top3voisins_precision, decimals=2)
        fig = plt.figure(figsize=(20, 20))
        plt.imshow(matrice_confusion_prct)
        plt.title(
            f"Matrice de confusion de la classe {classe_index} {identifieur_nom}: {self.correspondances_index_classes[classe_index]} \nPrécision : {matrice_confusion_prct[-1, -1]:.2f} % ; Top-3 précision : {top3voisins_precision:.2f} %")
        ax = plt.gca()
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.set_xlabel("Valeurs prédites")
        ax.set_ylabel("Valeurs vraies")
        ax.set_xticks(ticks=list(range(len(labels))))
        ax.set_yticks(ticks=list(range(len(labels))))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for x, y in np.ndindex(nb_val + 1, nb_val + 1):
            if (x < nb_val and y < nb_val) or x == y:
                plt.text(y, x, "    %.2f%%    \n    %d     " % (matrice_confusion_prct[x, y], matrice_confusion[x, y]),
                         ha="center", va="center", color="red")
        plt.tight_layout()
        plt.colorbar()
        plt.set_cmap('Blues')
        # get image in numpy array (thanks to https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = np.stack((data,), axis=0)
        if self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.image(f"matrice_confusion_{self.correspondances_index_classes[classe_index]}_{identifieur_nom}", data, step=0)
                self.summary_writer.flush()
        array_matrice = np.array([['' for _ in range(len(labels) + 2)] for _ in range(len(labels) + 3)], dtype="U25")
        array_matrice[0, :] = np.array(['Valeurs predites'] + [str(i) for i in labels] + [''], dtype=np.str)
        array_matrice[1, 0] = "Valeurs reelles"
        array_matrice[2:-1, 0] = np.array([str(i) for i in labels], dtype=np.str)
        array_matrice[2:-1, 1:-1] = addStr(
            addStr(
                addStr(np.array(matrice_confusion_petite, dtype=np.str), " | "),
                np.array(np.round(matrice_confusion_prct_petite, decimals=2), dtype=np.str)
            ), "%"
        )
        array_matrice[-1, -1] = str(matrice_confusion[-1, -1]) + "\n" + str(
            np.round(matrice_confusion_prct[-1, -1], decimals=2)) + "%"
        df = pd.DataFrame(array_matrice)
        df.to_csv(FolderInfos.base_filename + f"matrice_confusion_{self.correspondances_index_classes[classe_index]}_{identifieur_nom}.csv")
    def __call__(self):
        # Constitution de la matrice de confusion finale
        self.matrices_confusion = [{"true":[],"pred":[]} for _ in range(self.nb_classes)]
        for batch_img, batch_true in self.dataset:
            self.add_sample(batch_img,batch_true)
        # Création des matrices de confusion (1 par classe)
        for i,classe_data in enumerate(self.matrices_confusion):
            labels = list({val for val in (classe_data["true"] + classe_data["pred"])})
            matrice_confusion_petite = confusion_matrix(y_true=classe_data["true"],
                                                 y_pred=classe_data["pred"],
                                                 labels=labels)
            self.save_confusion_matrix(matrice_confusion_petite,labels,i)


if __name__ == "__main__":
    MakeConfusionMatrix(None,[(np.random.randint(0,1,(10,255,255,3)),np.random.choice([i for i in range(100) if i<10 or i > 90],(10,1,23)))
                        for _ in range(5000)],nb_classes=23,
                  correspondances_index_classes={0: 'animal',
                                                 1: 'human.pedestrian.adult',
                                                 2: 'human.pedestrian.child',
                                                 3: 'human.pedestrian.construction_worker',
                                                 4: 'human.pedestrian.personal_mobility',
                                                 5: 'human.pedestrian.police_officer',
                                                 6: 'human.pedestrian.stroller',
                                                 7: 'human.pedestrian.wheelchair',
                                                 8: 'movable_object.barrier',
                                                 9: 'movable_object.debris',
                                                 10: 'movable_object.pushable_pullable',
                                                 11: 'movable_object.trafficcone',
                                                 12: 'static_object.bicycle_rack',
                                                 13: 'vehicle.bicycle',
                                                 14: 'vehicle.bus.bendy',
                                                 15: 'vehicle.bus.rigid',
                                                 16: 'vehicle.car',
                                                 17: 'vehicle.construction',
                                                 18: 'vehicle.emergency.ambulance',
                                                 19: 'vehicle.emergency.police',
                                                 20: 'vehicle.motorcycle',
                                                 21: 'vehicle.trailer',
                                                 22: 'vehicle.truck'},summary_writer=None)()
    plt.show()





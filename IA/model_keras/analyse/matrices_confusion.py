from typing import Dict, List
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf

def make_matrices(model,dataset,nb_classes,correspondances_index_classes,summary_writer=None):
        # Constitution de la matrice de confusion finale
        matrices_confusion = [{"true":[],"pred":[]} for _ in range(nb_classes)]
        for batch_img, batch_true in dataset:
            # batch_pred = np.random.choice([i for i in range(100) if i<10 or i > 90],batch_true.shape) # Pour test
            batch_pred = model.predict(batch_img)
            batch_true = np.array(np.round(batch_true[:,0,:].numpy()),dtype=np.int)
            # batch_true = np.array(np.round(batch_true[:,0,:]),dtype=np.int)
            for i_batch in range(len(batch_pred)):
                for i_classe in range(nb_classes):
                    matrices_confusion[i_classe]["true"].append(batch_true[i_batch,i_classe])
                    matrices_confusion[i_classe]["pred"].append(batch_pred[i_batch, i_classe])
        # Création des matrices de confusion (1 par classe)
        for i,classe_data in enumerate(matrices_confusion):
            labels = list({val for val in (classe_data["true"] + classe_data["pred"])})
            matrice_confusion_petite = confusion_matrix(y_true=classe_data["true"],
                                                 y_pred=classe_data["pred"],
                                                 labels=labels)
            total = np.sum(matrice_confusion_petite)
            nb_val = len(labels)
            matrice_confusion_prct_petite = matrice_confusion_petite / total * 100
            matrice_confusion_prct = np.zeros((nb_val+1,nb_val+1))
            matrice_confusion_prct[:nb_val,:nb_val] = matrice_confusion_prct_petite
            matrice_confusion_prct[nb_val,nb_val] = np.sum(np.diag(matrice_confusion_prct_petite))

            matrice_confusion = np.zeros((nb_val+1,nb_val+1))
            matrice_confusion[:nb_val,:nb_val] = matrice_confusion_petite
            matrice_confusion[nb_val,nb_val] = np.sum(np.diag(matrice_confusion_petite))
            top3voisins_precision = np.sum(np.diag(matrice_confusion_prct_petite)) \
                                    + np.sum(np.diag(matrice_confusion_prct_petite,k=-1)) \
                                    + np.sum(np.diag(matrice_confusion_prct_petite,k=1))
            fig = plt.figure(figsize=(20,20))
            plt.imshow(matrice_confusion_prct)
            plt.title(f"Matrice de confusion de la classe {i} : {correspondances_index_classes[i]} \nPrécision : {matrice_confusion_prct[-1,-1]:.2f} % ; Top-3 précision : {top3voisins_precision:.2f} %")
            ax = plt.gca()
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            ax.set_xlabel("Valeurs prédites")
            ax.set_ylabel("Valeurs vraies")
            ax.set_xticks(ticks=list(range(len(labels))))
            ax.set_yticks(ticks=list(range(len(labels))))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            for x, y in np.ndindex(nb_val+1, nb_val+1):
                if (x < nb_val and y < nb_val) or x==y:
                    plt.text(y,x, "    %.2f%%    \n    %d     " % (matrice_confusion_prct[x, y],matrice_confusion[x, y]),
                       ha="center", va="center", color="red")
            plt.tight_layout()
            plt.colorbar()
            plt.set_cmap('Blues')
            # get image in numpy array (thanks to https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data = np.stack((data,), axis=0)
            if summary_writer is not None:
                with summary_writer.as_default():
                    tf.summary.image(f"matrice_confusion_{correspondances_index_classes[i]}", data, step=0)
                    summary_writer.flush()

if __name__ == "__main__":
    make_matrices(None,[(np.random.randint(0,1,(10,255,255,3)),np.random.choice([i for i in range(100) if i<10 or i > 90],(10,1,23)))
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
                                                 22: 'vehicle.truck'},summary_writer=None)
    plt.show()





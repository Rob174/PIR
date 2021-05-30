from IA.model_keras.analyse.matrices_confusion import MakeConfusionMatrix
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2

class MakeConfusionMatrixEnet(MakeConfusionMatrix):
    """Utilise le modèle de segmentation tel quel et "crée" les objets en seuillant la prédiction :
    Toute zone où la probabilité que le pixel appartienne à une classe dépasse seuil_threshold sera considéré comme appartenant à cette classe
    On peut ensuite détecter les contours et compter le nombre d'objets"""


    def __init__(self,seuils_threshold,*args,**kargs):
        super(MakeConfusionMatrixEnet, self).__init__(*args,**kargs)
        self.seuils_threshold = seuils_threshold
    def add_sample(self,batch_img, batch_true):
        batch_pred = self.model.predict(batch_img)
        batch_true = np.array(batch_true.numpy(), dtype=np.float32)
        for i_batch in range(len(batch_pred)):
            for i_classe in range(self.nb_classes):
                for batch, dest in zip([batch_true,batch_pred],["true","pred"]):
                    for seuil_threshold in self.seuils_threshold:
                        _,img_threshold = cv2.threshold(batch[i_batch,:,:, i_classe],
                                                      seuil_threshold,
                                                      maxval=1.,
                                                      type=cv2.THRESH_BINARY)
                        img_threshold = np.array(img_threshold*255,dtype=np.uint8)
                        nb_contours = len(cv2.findContours(img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])
                        self.matrices_confusion[i_classe][dest][seuil_threshold].append(nb_contours)
    def __call__(self):
        # Constitution de la matrice de confusion finale
        self.matrices_confusion = [{"true": {k:[] for k in self.seuils_threshold}, "pred": {k:[] for k in self.seuils_threshold}} for _ in range(self.nb_classes)]
        for batch_img, batch_true in self.dataset:
            self.add_sample(batch_img, batch_true)
        # Création des matrices de confusion (1 par classe)
        for i, classe_data in enumerate(self.matrices_confusion):
            for seuil_threshold in self.seuils_threshold:
                labels = list({val for val in (classe_data["true"][seuil_threshold] + classe_data["pred"][seuil_threshold])})
                matrice_confusion_petite = confusion_matrix(y_true=classe_data["true"][seuil_threshold],
                                                            y_pred=classe_data["pred"][seuil_threshold],
                                                            labels=labels)
                self.save_confusion_matrix(matrice_confusion_petite, labels,i,identifieur_nom=f"seuil-{seuil_threshold}")
from IA.model_keras.analyse.matrices_confusion import MakeConfusionMatrix
import numpy as np
import cv2

class MakeConfusionMatrixEnet(MakeConfusionMatrix):
    """Utilise le modèle de segmentation tel quel et "crée" les objets en seuillant la prédiction :
    Toute zone où la probabilité que le pixel appartienne à une classe dépasse seuil_threshold sera considéré comme appartenant à cette classe
    On peut ensuite détecter les contours et compter le nombre d'objets"""


    def __init__(self,seuil_threshold,*args,**kargs):
        super(MakeConfusionMatrixEnet, self).__init__(*args,**kargs)
        self.seuil_threshold = seuil_threshold
    def add_sample(self,batch_img, batch_true):
        batch_pred = self.model.predict(batch_img)
        batch_true = np.array(batch_true.numpy(), dtype=np.float32)
        for i_batch in range(len(batch_pred)):
            for i_classe in range(self.nb_classes):
                for batch, dest in zip([batch_true,batch_pred],["true","pred"]):
                    img_threshold = cv2.threshold(batch[i_batch,:,:, i_classe],
                                                  self.seuil_threshold,
                                                  maxval=1.,
                                                  type=cv2.THRESH_BINARY)
                    img_threshold = np.array(img_threshold,dtype=np.uint8)
                    nb_contours = len(cv2.findContours(img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))
                    self.matrices_confusion[i_classe][dest].append(nb_contours)
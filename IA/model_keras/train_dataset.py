from data.generate_data import Nexet_dataset
from model.model import make_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

dataset = Nexet_dataset()
liste_loss=[]
liste_accuracy=[]
model = make_model((900, 1600,3), num_classes=len(dataset.correspondances_classes.keys()))
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])
for epochs in range(1):
    iterator = dataset.getNextBatch()
    compteur=0
    while True:
        try:
            batchImg, batchLabel = next(iterator)
            print(batchImg.shape, batchLabel.shape)
            [loss,accuracy]=model.train_on_batch(batchImg,batchLabel)
            liste_loss.append(loss)
            liste_accuracy.append(accuracy)
            compteur=compteur+1
            if compteur==10:
                break
        except StopIteration:
            print("Epoch %d done" % epochs)
            break
plt.plot(liste_loss,color="r")
plt.plot(liste_accuracy, color="g")
plt.savefig("/home/acalmet/Documents/erreur_accuracy.png")
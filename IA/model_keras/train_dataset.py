from data.generate_data import Nexet_dataset
from model.model import make_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
      tf.config.experimental.set_memory_growth(device, True)
    except:
      # Invalid device or cannot modify virtual devices once initialized.
      pass
dataset = Nexet_dataset()
liste_loss=[]
liste_accuracy=[]
model = make_model((dataset.image_shape[1], dataset.image_shape[0],3), num_classes=len(dataset.correspondances_classes.keys()))
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])
for epochs in range(1):
    iterator = dataset.getNextBatch()
    compteur=0
    while True:
        try:
            batchImg, batchLabel = next(iterator)
            batchLabel = np.argmax(batchLabel,axis=1)
            [loss,accuracy]=model.train_on_batch(batchImg,batchLabel)
            liste_loss.append(loss)
            liste_accuracy.append(accuracy)
            compteur=compteur+1
            if compteur==10:
                break
        except StopIteration:
            print("Epoch %d done" % epochs)
            break
fig, axe_error = plt.subplots()
loss_axe = axe_error.twinx()
loss_axe.plot(np.arange(0,len(liste_loss)*dataset.batch_size,dataset.batch_size),liste_loss,color="r",label="loss")
axe_error.plot(np.arange(0,len(liste_accuracy)*dataset.batch_size,dataset.batch_size),100*(1-np.array(liste_accuracy)),
               color="g",label="tr_error")
axe_error.set_xlabel("Nombre d'itérations (nb de batch parcourus/lots d'images)")
axe_error.set_ylabel("Error (%)")
loss_axe.set_ylabel("Loss (sparsecategoricalcrossentropy)")
fig.legend()
plt.grid()
plt.savefig("/home/rmoine/Documents/erreur_accuracy.png")
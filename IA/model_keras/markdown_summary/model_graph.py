import tensorflow as tf
from PIL import Image
import numpy as np

if __name__ == "__main__":
    folder = "C:/Users/robin/Documents/projets/PIR/data/2021-04-19_16h46min23s_/"

    file_writer = tf.summary.create_file_writer(folder)
    file_writer.set_as_default()

    with file_writer.as_default():
        image = np.array(Image.open(folder+"2021-04-19_16h46min23s_model.png"))
        image = np.expand_dims(image,axis=0)
        print(image.shape)
        tf.summary.image("Modele",image,step=0)
        file_writer.flush()
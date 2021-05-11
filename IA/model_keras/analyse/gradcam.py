import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.cm as cm
import cv2
import matplotlib.pyplot as plt

from IA.model_keras.FolderInfos import FolderInfos
from IA.model_keras.data.Nuscene_dataset import Nuscene_dataset


class GradCam:
    def __init__(self, model,last_conv_layer_name,summary_writer):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        self.grad_model = Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        self.summary_writer = summary_writer
    def __call__(self, img_array, pred_index=None,identifieur="",*args, **kwargs):
        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        superimposed_img = self.superimposed_img(img_array,heatmap)
        plt.clf()
        fig = plt.figure()
        plt.title(f"Gradcam de la classe {Nuscene_dataset.correspondances_index_classes[pred_index]} {identifieur} avec {class_channel[0,pred_index]} elements")
        plt.imshow(superimposed_img)
        plt.savefig(FolderInfos.base_filename+f"gradcam_{Nuscene_dataset.correspondances_index_classes[pred_index]}_{identifieur}")

        # Save the superimposed image
        # get image in numpy array (thanks to https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = np.stack((data,), axis=0)
        if self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.image(
                    f"gradcam_{Nuscene_dataset.correspondances_index_classes[pred_index]}_{identifieur}", data, step=0)
                self.summary_writer.flush()

        return heatmap.numpy()
    def superimposed_img(self,img,grad_img):
        grad_img = (grad_img * 255).astype(np.uint8)
        img = (img*255).astype(np.uint8)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[grad_img]

        # Create an image with RGB colorized heatmap
        jet_heatmap = cv2.resize(jet_heatmap,dsize=img.shape[:-1],interpolation=cv2.INTER_LANCZOS4)

        # Superimpose the heatmap on original image
        superimposed_img = (jet_heatmap * 0.2 + img *255* 0.8).astype(np.unint8)
        return superimposed_img

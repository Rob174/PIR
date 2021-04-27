import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Luminance_augment:
    def __init__(self):
        pass
    @staticmethod
    def augment(image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_hsv[:,:,2] *= (0.2+1.5*np.random.rand())
        image_hsv[:,:,2] = np.clip(image_hsv[:,:,2],0,255)
        image = cv2.cvtColor(image_hsv,cv2.COLOR_HSV2RGB)
        return image

if __name__ == "__main__":
    path = r"C:\Users\robin\Documents\projets\nuscene\samples\CAM_BACK\n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg"
    image = Image.open(path)
    image = np.array(image.resize((400,225)),dtype=np.float32)
    print(np.max(image))
    plt.figure(1)
    plt.imshow(image/255.)
    image_augm = Luminance_augment.augment(image)
    print(np.max(image_augm))
    plt.figure(2)
    plt.imshow(image_augm/255.)
    plt.show()
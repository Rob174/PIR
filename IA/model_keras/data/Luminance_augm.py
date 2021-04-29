import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Luminance_augment:
    augm_params = {
    }
    @staticmethod
    def augment(image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_hsv[:,:,2] *= (0.2+1.5*np.random.rand())
        image_hsv[:,:,2] = np.clip(image_hsv[:,:,2],0,1.)
        image = cv2.cvtColor(image_hsv,cv2.COLOR_HSV2RGB)
        image = np.array(image,dtype=np.float)
        return image

if __name__ == "__main__":
    path = r"C:\Users\robin\Documents\projets\nuscene\samples\CAM_BACK\n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg"
    image = Image.open(path)
    image = np.array(image.resize((400,225)),dtype=np.float32)
    image /= 255.
    plt.figure(1)
    plt.imshow(image)
    image_augm = Luminance_augment.augment(image)
    plt.figure(2)
    plt.imshow(image_augm)
    # plt.imshow(cv2.Canny(cv2.cvtColor(np.array(image*255,dtype=np.uint8), cv2.COLOR_BGR2GRAY),50,50),cmap='gray')

    plt.show()
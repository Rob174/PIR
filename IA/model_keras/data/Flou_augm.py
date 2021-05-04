import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class Flou_augment:
    augm_params = {
        "max_std_flou":2
    }
    @staticmethod
    def augment(image):
        std_flou = np.random.rand()*Flou_augment.augm_params["max_std_flou"]
        for c in range(3):
            image[:, :, c] = gaussian_filter(image[:, :, c], sigma=(std_flou, std_flou))
        return image

if __name__ == "__main__":
    path = r"C:\Users\robin\Documents\projets\nuscene\samples\CAM_BACK\n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg"
    image = Image.open(path)
    image = np.array(image.resize((400,225)),dtype=np.float32)
    image /= 255.
    print(np.max(image))
    plt.figure(1)
    plt.imshow(image)
    image_augm = Flou_augment.augment(image)
    print(np.max(image_augm))
    plt.figure(2)
    plt.imshow(image_augm)
    plt.show()
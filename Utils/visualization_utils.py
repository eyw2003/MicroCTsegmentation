import matplotlib.pyplot as plt
from Utils.dataset_utils import normalize
import cv2
import numpy as np
def draw_mask(image,mask=None,color=(255,0,0)):
    image = normalize(image)
    image = image * 255
    image = image.astype('uint8')
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    rgb[np.where(mask==1)]=color
    return rgb


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 16))

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image,cmap="gray")
    plt.show()
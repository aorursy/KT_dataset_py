import cv2

import os

import numpy as np

from PIL import Image, ImageEnhance

from matplotlib import pyplot as plt

path = '../input/gtsrb-german-traffic-sign/train/1/00001_00003_00021.png'



img = cv2.imread(path,0)

edges = cv2.Canny(img, 250, 500)



image=cv2.imread(path, 0)

pic = Image.fromarray(edges)



plt.subplot(121),plt.imshow(image,cmap = 'gray')

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(edges,cmap = 'gray')

plt.title('Edge Image'), plt.xticks([]), plt.yticks([])



plt.show()
h = 250

w = 250

image = Image.open(path)

enh = ImageEnhance.Brightness(image)

enh_img = enh.enhance(2)

eh_edges = cv2.Canny(np.array(enh_img), 0, 150)

or_image = cv2.Canny(np.array(image), 50, 250)



display(image.resize((h, w)))

display(enh_img.resize((h, w)))



plt.subplot(121),plt.imshow(eh_edges,cmap = 'gray')

plt.title('Eh ED Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(or_image,cmap = 'gray')

plt.title('Ori ED Image'), plt.xticks([]), plt.yticks([])
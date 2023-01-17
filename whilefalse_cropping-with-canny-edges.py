
import numpy as np # linear algebra
from PIL import Image
import cv2
import os
# the algorithm to crop an image 
# first crop some edges
# second denoise the image and use canny edge to find the object
# set some threshold. If below some value, then abandon that area
# the last step is to prevent the low quality of image making the edge too sparse.
def crop(img):
    origi = img.copy()
    img = img.copy()
    if img.size[1] > 300:
        img = img.crop((20, 20, img.size[0]-40, img.size[1]-80))

    im = np.array(img)

    im = cv2.GaussianBlur(im, (3, 3), 0)
    a = cv2.Canny(im, 150, 200)

    k1 = 0
    for j in range(int(img.size[1]*2/3)):
        if np.mean(a[j]) < a.shape[1] / 300 or np.mean(a[j]) > len(a[j])*2/5:
            k1 += 1
        else:
            break

    k2 = 0
    for j in range(1, int(img.size[1]*2/3)):
        if np.mean(a[img.size[1]-j]) < a.shape[1] / 300 or np.mean(a[img.size[1]-j]) > len(a[img.size[1]-j])*2/5:
            k2 += 1
        else:
            break

    k3 = 0
    for j in range(int(img.size[0]*2/3)):
        if np.mean(a[:, j]) < a.shape[0] / 300 or np.mean(a[:, j]) > len(a[j]) * 2 / 5.5:
            k3 += 1
        else:
            break

    k4 = 0
    for j in range(1, int(img.size[0]*2/3)):
        if np.mean(a[:, img.size[0]-j]) < a.shape[0] / 300 or np.mean(a[:, img.size[0]-j]) > len(a[:, img.size[0]-j])*2/5:
            k4 += 1
        else:
            break
    img = img.crop((k3, k1, img.size[0]-k4, img.size[1]-k2))
    

    if img.size[0]*img.size[1] > 10000 and img.size[1]/img.size[0] < 6:
        return img
    else:
        return origi
# see some results, obviously not perfect
for i in range(220, 230):
    import matplotlib.pyplot as plt
    im = Image.open('../input/train/'+os.listdir('../input/train/')[i]).convert('RGB')
    orange = np.array(im)
    
    fig=plt.figure()
    one = fig.add_subplot(121)
    two = fig.add_subplot(122)
    one.imshow(orange)
    new = np.array(crop(im))
    two.imshow(new)
    plt.show()
    
    
# hope someone could find better hyperparameters
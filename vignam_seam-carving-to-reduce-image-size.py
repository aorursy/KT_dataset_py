import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from skimage.filters import sobel,gaussian
from skimage.transform import seam_carve 
from keras.preprocessing.image import load_img
trainset = pd.read_csv("../input/train.csv", index_col="Id")
ix = 200
filters = ['green','blue','red','yellow']
image = np.sum([np.array(load_img('../input/train/{}_{}.png'.format(trainset.index[ix],k)))[:,:,0]/255 for k in filters],axis=0)
eimage = sobel(gaussian(image,4))
imageh = seam_carve(image, eimage,'horizontal',100)
eimageh = sobel(gaussian(imageh,4))
finimage = seam_carve(imageh, eimageh,'vertical',100)
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(finimage)

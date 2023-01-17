# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import fastai
fastai.__version__  # 2.0.15
# https://towardsdatascience.com/deep-learning-image-classification-with-fast-ai-fc4dc9052106
from fastai.vision.all import *
set_seed(2)

from fastai.metrics import error_rate, accuracy, RocAuc, CohenKappa, RocAucBinary
import warnings
warnings.filterwarnings('ignore')
path = '../input/chest-xray-pneumonia/chest_xray'
# https://medium.com/hackernoon/histogram-equalization-in-python-from-scratch-ebb9c8aa3f23
%matplotlib inline
from IPython.display import display, Math, Latex

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# create our own histogram function
def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)
    
    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1
    
    # return our final result
    return histogram

# create our cumulative sum function
def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

img = Image.open('{}/train/NORMAL/IM-0115-0001.jpeg'.format(path))

# display the image
# plt.imshow(img, cmap='gray')

# convert our image into a numpy array
imgnp = np.asarray(img)

# put pixels in a 1D array by flattening out img array
flat = imgnp.flatten()

# show the histogram
# plt.hist(flat, bins=50)

# execute our histogram function
hist = get_histogram(flat, 256)

# execute the fn
cs = cumsum(hist)

# display the result
# plt.plot(cs)

# numerator & denomenator
nj = (cs - cs.min()) * 255
N = cs.max() - cs.min()

# re-normalize the cumsum
cs = nj / N

# cast it back to uint8 since we can't use floating point values in images
cs = cs.astype('uint8')
# plt.plot(cs)

# get the value from cumulative sum for every index in flat, and set that as img_new
img_new = cs[flat]

# put array back into original shape since we flattened it
img_new = np.reshape(img_new, imgnp.shape)

# set up side-by-side image display
fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)

fig.add_subplot(1,2,1)
plt.imshow(imgnp, cmap='gray')

# display the new image
fig.add_subplot(1,2,2)
plt.imshow(img_new, cmap='gray')

plt.show(block=True)
import PIL
from fastai.vision.core import PILImage
# Parts taken from: 
# https://medium.com/hackernoon/histogram-equalization-in-python-from-scratch-ebb9c8aa3f23
# https://medium.com/@pierre_guillou/2-2-fastai-the-new-radiology-tool-9f0b7db7bf91



class HistogramEqualization(Transform):
    def __init__(self, prefix=None):
        self.prefix = prefix or ""

    def encodes(self, o):
        # convert our image into a numpy array
        imgnp = o.cpu().numpy()  # np.asarray(o[0]) # .permute(1, 2, 0))
        
        # put pixels in a 1D array by flattening out img array
        flat = imgnp.flatten()
        # execute our histogram function
        hist = get_histogram(flat, 256)

        # execute the fn
        cs = cumsum(hist)
        # numerator & denomenator
        nj = (cs - cs.min()) * 255
        N = cs.max() - cs.min()

        # re-normalize the cumsum
        cs = nj / N

        # cast it back to uint8 since we can't use floating point values in images
        cs = cs.astype('uint8')

        # get the value from cumulative sum for every index in flat, and set that as img_new
        img_new = cs[flat]

        # put array back into original shape since we flattened it
        img_new = np.reshape(img_new, o.shape)

        ret = TensorImage(img_new) if (type(o) == TensorImage) else o
        return ret
        
    def decodes(self, o):
        return o
    
class HistogramEqualization_item(Transform):
    def __init__(self, prefix=None):
        self.prefix = prefix or ""

    def encodes(self, o):
#         print(type(o))
        if type(o) == PILImage:
            ret = PIL.ImageOps.equalize(o)  
        else:
            ret = o
        return ret
    
    def decodes(self, o):
        return o
data = ImageDataLoaders.from_folder(path, train='train', valid='test',
                                    item_tfms=[Resize(size=384)],#, HistogramEqualization_item()],
                                    batch_tfms=[*aug_transforms(min_scale=0.98, do_flip=False)], 
                                    max_zoom=1.1, max_lighting=0.2, bs=16, num_workers=8) 
# Show what the data looks like after being transformed
data.show_batch()
# See the classes and count of classes in your dataset
print(data.c)
# See the number of images in each data set
print(len(data.train_ds), len(data.valid_ds))
learn = cnn_learner(data, densenet169, metrics=RocAucBinary()).to_fp16()
from pathlib import Path
learn.path = Path('/kaggle/working/')
# Augmentation of the weight decay to wd=0.1 and use of the callback function SaveModelCallback() in order to save the model after an epoch if its kappa score is the biggest.
learn.fit_one_cycle(1, wd=0.1)
# Save the model
learn.save('densenet169-auc-stage-1')
# Load the Model
learn.load('densenet169-auc-stage-1')
# Unfreeze all layers of the CNN
learn.unfreeze()
# Find the optimal learning rate and plot a visual
suggested = learn.lr_find()
# Fit the model over 4 epochs
learn.unfreeze()
learn.fit_one_cycle(4, lr_min=suggested.lr_min, lr_steep=suggested.lr_steep, wd=0.1)
# Save the model
learn.save('densenet169-auc-stage-2')
# Load the Model
learn.load('densenet169-auc-stage-2')
learn.fit_one_cycle(4, lr_min=suggested.lr_min, lr_steep=suggested.lr_steep, wd=0.1)
# Rebuild interpreter and replot confusion matrix
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.plot_top_losses(8)

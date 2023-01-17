# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.ndimage.filters as fi
import keras
from keras.models import *
from keras.layers import *
import matplotlib.pyplot as plt
%matplotlib inline
data = np.load("../input/mnist.npz")
def get_projections(img):
    proj_cols = []
    proj_rows = []
    proj_diag = []
    for i in range(img.shape[0]):
        proj_rows.append(np.sum(img[i,:]))
        
    for i in range(img.shape[0]):
        proj_cols.append(np.sum(img[:, i]))
        
    for i in range(img.shape[0]):
        proj_diag.append(np.sum(np.diagonal(img,i)))
        
    return proj_rows, proj_cols, proj_diag


X = []
Y = []
for img in data['x_test']:
    img = fi.gaussian_filter(img, 1)
    proj = get_projections(img)
    X.append(proj[0] + proj[1] + proj[2])
    Y.append(img)
    
# Convert X and Y to numpy arrays
X = np.asarray(X)
Y = np.asarray(Y)

# Normalize
X = (X - np.average(X)) / np.std(X)
Y = (Y - np.average(Y)) / np.std(Y)

print("X: ", X.shape)
print("Y: ", Y.shape)
def restr_model():
    return keras.models.Sequential([
        Dense(49, input_shape=(84,)),
        Activation('relu'),
        Reshape(target_shape=(7,7,1)),
        Deconvolution2D(64, 5),
        UpSampling2D(size=(2, 2), interpolation="bilinear"),
        Deconvolution2D(32, 3),
        Deconvolution2D(16, 3),
        Deconvolution2D(8, 3),
        Deconvolution2D(1, 3, padding="same"),
        Reshape(target_shape=(28,28))
    ])
    
model = restr_model()
model.compile(optimizer='rmsprop', loss='mse')
model.summary()
model.fit(X, Y, validation_split=0.2, epochs=100)
preds = model.predict(X)
# X[0]
def show_side_by_side(sample_idx):
    f, axes = plt.subplots(2, len(sample_idx), sharey=True, figsize=(2*len(sample_idx), 3))
    for i in range(len(sample_idx)):
        axes[0, i].imshow(Y[sample_idx[i]], cmap="gray_r")
        axes[1, i].imshow(preds[sample_idx[i]], cmap="gray_r")

show_side_by_side([6942, 1252, 3293, 129, 9, 9992])
diff = np.std((preds - Y), axis=(1,2))
_ = plt.hist(diff)
diff_idx = np.arange(len(diff))
sorted_diffs = sorted(zip(diff, diff_idx), key=lambda d: d[0])
show_side_by_side([d[1] for d in sorted_diffs[-10:]])
show_side_by_side([d[1] for d in sorted_diffs[900:910]])

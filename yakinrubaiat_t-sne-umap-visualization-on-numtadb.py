# -*- coding: utf-8 -*-



# I don't like warnings, especially user warnings at all!

import warnings

warnings.filterwarnings('ignore')
# Import some packages that we require

import os

import glob

import umap

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

from PIL import Image

from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential, Model

from keras.applications import vgg16

from keras.applications import resnet50

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.optimizers import Adam, RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

from skimage.io import imread, imshow

from skimage.transform import resize

import imgaug as ia

from imgaug import augmenters as iaa

from keras import backend as K

import tensorflow as tf

from collections import defaultdict, Counter

from sklearn.preprocessing import StandardScaler

print(os.listdir("../input"))
# For plotting within the notebook

%matplotlib inline



# Graphics in SVG format are more sharp and legible

%config InlineBackend.figure_format = 'svg'



# seaborn color palette 

color = sns.color_palette()



# For REPRODUCIBILITY

seed = 17

np.random.seed(seed)

tf.set_random_seed(seed)
input_path = Path("../input")



# Path to training images and corresponding labels provided as numpy arrays

numtadb_train_path = input_path/"NumthDB_training.npz"



# Path to the test images and corresponding labels

numtadb_test_path = input_path/"NumthDB_test.npz"
train_images = np.load(numtadb_train_path)['data']

train_labels = np.load(numtadb_train_path)['label']



# Load the test data from the corresponding npz files

test_images = np.load(numtadb_test_path)['data']
train_images[0]
plt.imshow(train_images[1]);
print(f"Number of training samples: {len(train_images)} where each sample is of size: {train_images.shape[1:]}")

print(f"Number of test samples: {len(test_images)} where each sample is of size: {test_images.shape[1:]}")
# Get the unique labels

labels = np.unique(train_labels)



# Get the frequency count for each label

frequency_count = np.bincount(train_labels)



# Visualize 

plt.figure(figsize=(10,5))

sns.barplot(x=labels, y=frequency_count);

plt.title("Distribution of labels in NumtaDB training data", fontsize=16)

plt.xlabel("Labels", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.show()
random_samples = []

for i in range(10):

    samples = train_images[np.where(train_labels==i)][:3]

    random_samples.append(samples)



# Converting list into a numpy array

random_samples = np.array(random_samples)



# Visualize the samples

f, ax = plt.subplots(10,3, figsize=(10,20))

for i, j in enumerate(random_samples):

    ax[i, 0].imshow(random_samples[i][0,:,:], cmap='gray')

    ax[i, 1].imshow(random_samples[i][1,:,:], cmap='gray')

    ax[i, 2].imshow(random_samples[i][2,:,:], cmap='gray')

    

    ax[i,0].set_title(str(i))

    ax[i,0].axis('off')

    ax[i,0].set_aspect('equal')

    

    ax[i,1].set_title(str(i))

    ax[i,1].axis('off')

    ax[i,1].set_aspect('equal')

    

    ax[i,2].set_title(str(i))

    ax[i,2].axis('off')

    ax[i,2].set_aspect('equal')

plt.show()
def get_random_samples(nb_indices):

    # Choose indices randomly 

    random_indices = np.random.choice(nb_indices, size=nb_indices, replace=False)



    # Get the data corresponding to these indices

    random_train_images = train_images[random_indices].astype(np.float32)

    random_train_images /=255.

    random_train_images = random_train_images.reshape(nb_indices, 28*28)

    random_train_labels = train_labels[random_indices]

    labels = np.unique(random_train_labels)

    return random_indices, random_train_images, random_train_labels, labels
#Get randomly sampled data

nb_indices = 20000

random_indices, random_train_images, random_train_labels, labels = get_random_samples(nb_indices)



# Get the actual labels from the labels dictionary

labels_name = [x for x in labels]



# Get a t-SNE instance

tsne = TSNE(n_components=2, random_state=seed, perplexity=60)



# Do the scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(random_train_images)



# Fit tsne to the data

random_train_2D = tsne.fit_transform(X_scaled)
fig = plt.figure(figsize=(10, 8))



for i, label in zip(labels, labels_name):

    sns.scatterplot(random_train_2D[random_train_labels == i, 0], 

                random_train_2D[random_train_labels == i, 1], 

                label=i, s=18)



plt.title("Visualizating NumtaDb embeddings using tSNE", fontsize=16)

plt.legend()

plt.show()

plt.savefig('t-SNE Visualization in NumtaDB');
# Let's try UMAP now.

nb_indices = 50000

random_indices, random_train_images, random_train_labels, labels = get_random_samples(nb_indices)



embedding = umap.UMAP(n_components=2, metric='correlation', min_dist=0.2)



scaler = StandardScaler()

X_scaled = scaler.fit_transform(random_train_images)



random_train_2D = embedding.fit_transform(X_scaled)
fig = plt.figure(figsize=(10, 8))



for i, label in zip(labels, labels):

    sns.scatterplot(random_train_2D[random_train_labels == i, 0], 

                random_train_2D[random_train_labels == i, 1], 

                label=label, s=15)

plt.title("Visualiza NumtaDb embeddings using UMAP ", fontsize=16)

plt.legend()

plt.show()

plt.savefig('UMap Visualization in NumtaDB');
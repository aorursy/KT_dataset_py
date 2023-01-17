import sys

!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cudf, cuml

import pandas as pd

import numpy as np

from cuml.manifold import TSNE

import matplotlib.pyplot as plt

%matplotlib inline
target = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_1/target.npy')
train = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_1/train_VGG16.npy')
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c = target, s = 0.5)
train = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_1/train_VGG19.npy')
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c = target, s = 0.5)
train = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_1/train_ResNet50.npy')
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c = target, s = 0.5)
train = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_1/train_InceptionV3.npy')
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c = target, s = 0.5)
train = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_1/train_InceptionResNetV2.npy')
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c = target, s = 0.5)
train = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_2/train_ResNet152.npy')
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c = target, s = 0.5)
train = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_2/train_NAS_large.npy')
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c = target, s = 0.5)
train = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_3/cats_and_dogs_3/train_EB5.npy')
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c = target, s = 0.5)
train = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_3/cats_and_dogs_3/train_EB6.npy')
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c = target, s = 0.5)
train = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_4/cats_and_dogs_4/train_EB4_ns.npy')
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c = target, s = 0.5)
train = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_4/cats_and_dogs_4/train_EB7_ns.npy')
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c = target, s = 0.5)
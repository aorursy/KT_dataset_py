import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import cv2 as cv

from re import match

from itertools import product, count, chain

from keras.utils import to_categorical



%matplotlib inline
images = os.listdir('../input/samples/samples')

images[0]
images = list(filter(lambda image: match('^[a-z0-9]+\..+$', image), images))
len(images)
texts = [match('^([a-z0-9]+)\..+$', image).group(1) for image in images]
all([len(text) == 5 for text in texts])
alphabet = list(frozenset(chain.from_iterable(texts)))

alphabet.sort()

''.join(alphabet)
ids = dict([(ch, alphabet.index(ch)) for ch in alphabet])

ids['b']
n, m = len(texts), 5

y_labels = np.zeros([n, m], dtype=np.uint8)

for i, j in product(range(0, n), range(0, m)):

    y_labels[i, j] = ids[texts[i][j]]

y_labels[0]
y = np.zeros([n, m, len(alphabet)], dtype=np.uint8)

for i, j in product(range(0, n), range(0, m)):

    y[i, j, :] = to_categorical(y_labels[i, j], len(alphabet))
y[0, 0, :]
y.shape
np.all((y_labels == y.argmax(axis=2)).flatten())
X = np.zeros((n,) + (50, 200, 1), dtype=np.float32)

for i, filename in zip(range(0, n), images):

    img = cv.cvtColor(cv.imread('../input/samples/samples/' + filename), cv.COLOR_BGR2GRAY)

    assert img.shape == (50, 200)

    X[i, :, :, 0] = img.astype(np.float32) / 255
plt.imshow(X[10, :, :, 0], cmap='gray'), plt.xticks([]), plt.yticks([]);
np.savez_compressed('preprocessed-data.npz', X=X, y=y, y_labels=y_labels, alphabet=alphabet)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import math, re, os,io
import tensorflow as tf
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn import model_selection
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
df = pd.read_csv('/kaggle/input/TrainData-C2/TrainAnnotations.csv')
filenames = [fname for fname in df['file_name'].tolist()]
labels =  [label for label in df['annotation'].tolist()]
print(df)
train_fnames,val_fnames,train_labels,val_labels = model_selection.train_test_split(filenames,labels,train_size=0.9,random_state=42)
BATCH_SIZE = 32
IMAGE_SIZE = 300
path = path = os.path.join("/kaggle/input/","TrainData-C2/")
train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(train_fnames),tf.constant(train_labels)))
val_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(val_fnames),tf.constant(val_labels)))
def load_image(image_name,label):
    image = tf.io.read_file(path+image_name)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image,label
#creating a dataset of images and labels pair
labeled_ds = train_dataset.map(load_image)
val_labeled_ds = val_dataset.map(load_image)
for image,label in labeled_ds.take(5):
    print("image shape:",np.array(image).shape)
    print("label shape:",label)
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=10000)

  return ds
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      #plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
train_ds = prepare_for_training(labeled_ds)
val_ds = prepare_for_training(val_labeled_ds)

image_batch, label_batch = next(iter(train_ds))
show_batch(image_batch.numpy(), label_batch.numpy())
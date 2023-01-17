# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle

!pip install opencv-python
import cv2
import numpy as np 
import pandas as pd 
import os
print(os.listdir("/kaggle/input/intel-image-classification/seg_train/seg_train"))

CNAMES = ['buildings', 'glacier', 'sea', 'mountain', 'forest', 'street']
_labels = {class_name:i for i, class_name in enumerate(CNAMES)}
_path = "/kaggle/input/intel-image-classification"

print(_labels)

IMGSIZE = (128, 128)
X_tr, y_tr, X_ts, y_ts = [], [], [], []
for label in _labels:
    print(label)
    path = _path + '/seg_train/seg_train/' + label
    for f in sorted([_ for _ in os.listdir(path) if _.lower().endswith('.jpg')]):
        X_tr += [cv2.resize(cv2.imread(os.path.join(path,f)), IMGSIZE)]
        y_tr += [CNAMES.index(label)]

X_tr = np.asarray(X_tr)
X_tr.shape[0]
y_tr = np.asarray(y_tr)
y_tr.shape[0]
X_tr, y_tr = shuffle(X_tr, y_tr, random_state=25)
y_tr[-1]
X_tr[-1]
def display_images(class_names, images, labels, num):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Here are some of the images :)", fontsize=16)
    for i in range(num):
        index = np.random.randint(images.shape[0])
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        
        #maps the RBG values of each picture to its respective color
        plt.imshow(images[index], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[index]])
    plt.show()

display_images(CNAMES, X_tr, y_tr, 10)
#current size of an image array...already at the correct size requirement
X_tr.shape
X_tr = X_tr / 255
X_tr
# Our full CNN neural network
cnn1 = tf.keras.Sequential()

cnn1.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
    data_format='channels_last',
    name='conv_1', activation='relu'))

cnn1.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_1'))

cnn1.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5),
    name='conv_2', activation='relu'))

cnn1.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_2'))

cnn1.add(tf.keras.layers.Flatten())

cnn1.add(tf.keras.layers.Dense(units=1024, name='fc_1', activation='relu'))

cnn1.add(tf.keras.layers.Dense(units=10, name='fc_2', activation='softmax'))
# Set a seed for repeatibility
tf.random.set_seed(0)

# Build the model
cnn1.build(input_shape=(None,128, 128, 3))

# Compile the model with the optimizer, loss function and metric
cnn1.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

NUM_EPOCHS = 8
# Save weights for debugging purposes and saving the model
cnn1.save_weights('cnn1_weights.h5')
%%time
history = cnn1.fit(X_tr, y_tr,
        epochs=NUM_EPOCHS,
        shuffle=True)
#cnn1.load_weights('cnn1_weights.h5')
for label in _labels:
    print(label)
    path = _path + '/seg_test/seg_test/' + label
    for f in sorted([_ for _ in os.listdir(path) if _.lower().endswith('.jpg')]):
        X_ts += [cv2.resize(cv2.imread(os.path.join(path,f)), IMGSIZE)]
        y_ts += [CNAMES.index(label)]
X_ts = np.asarray(X_ts)
X_ts = X_ts / 255
y_ts = np.asarray(y_ts)
X_ts
y_ts
y_ts.shape
y_pred = cnn1.predict_classes(X_ts)
print(f'Accuracy= {sum(y_pred==y_ts)/3000:.3f}')
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_ts, y_pred)
import seaborn as sns
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
print(_labels)
# Our full CNN neural network...with a dropout later
cnn2 = tf.keras.Sequential()

cnn2.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
    data_format='channels_last',
    name='conv_1', activation='relu'))

cnn2.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_1'))

cnn2.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5),
    name='conv_2', activation='relu'))

cnn2.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_2'))

cnn2.add(tf.keras.layers.Dropout(
    0.3, noise_shape=None, seed=None))

cnn2.add(tf.keras.layers.Flatten())

cnn2.add(tf.keras.layers.Dense(units=1024, name='fc_1', activation='relu'))

cnn2.add(tf.keras.layers.Dense(units=10, name='fc_2', activation='softmax'))
# Set a seed for repeatibility
tf.random.set_seed(0)

# Build the model
cnn2.build(input_shape=(None,128, 128, 3))

# Compile the model with the optimizer, loss function and metric
cnn2.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

NUM_EPOCHS = 8
# Save weights for debugging purposes and saving the model
cnn2.save_weights('cnn2_weights.h5')
%%time
history = cnn2.fit(X_tr, y_tr,
        epochs=NUM_EPOCHS,
        shuffle=True)
y_pred = cnn2.predict_classes(X_ts)
print(f'Accuracy= {sum(y_pred==y_ts)/3000:.3f}')

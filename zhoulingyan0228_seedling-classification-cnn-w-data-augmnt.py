import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
import sklearn
import sklearn.preprocessing
import skimage
import skimage.io
import skimage.color
import seaborn as sns
import random
import os
import cv2
import glob
import tensorflow as tf
TRAIN_DIR = '../input/train'
TEST_DIR = '../input/test'
IMG_HW = 256

labels_all = os.listdir(TRAIN_DIR)
N_CLASSES = len(labels_all)
labelEncoder = sklearn.preprocessing.LabelEncoder()
labelEncoder.fit(labels_all)

train_files = glob.glob(TRAIN_DIR+'/*/*.png')
train_labels = [f.split('/')[3] for f in train_files]
test_files = glob.glob(TEST_DIR+'/*.png')

train_files, train_labels = sklearn.utils.shuffle(train_files, train_labels)
train_labels_encoded = labelEncoder.transform(train_labels)
pd.Series(train_labels_encoded).plot.hist();
def jitter(img, max_jitter=25):
    pts1 = np.array(np.random.uniform(-max_jitter, max_jitter, size=(4,2))+np.array([[0,0],[0,IMG_HW],[IMG_HW,0],[IMG_HW,IMG_HW]])).astype(np.float32)
    pts2 = np.array([[0,0],[0,IMG_HW],[IMG_HW,0],[IMG_HW,IMG_HW]]).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,M,(IMG_HW,IMG_HW))

def rotate(img, rotation=None):
    if rotation == None:
        rotation = random.randint(0, 360)
    M = cv2.getRotationMatrix2D((IMG_HW/2,IMG_HW/2),90,1)
    return cv2.warpAffine(img,M,(IMG_HW,IMG_HW))
    
def resize(img):
    return cv2.resize(img, (IMG_HW, IMG_HW))

def train_generator(train_files, train_labels_encoded, augments = 20, img_per_batch=100):
    idx = 0
    maxIdx = len(train_files)
    while True:
        train_batch = []
        labels_batch = []
        for _ in range(img_per_batch):
            img = resize(skimage.io.imread(train_files[idx])/255.)
            if img.shape==(256,256,4):
                img = skimage.color.rgba2rgb(img)
            train_batch.append(img)
            labels_batch.append(train_labels_encoded[idx])
            for _ in range(augments):
                train_batch.append(rotate(jitter(img)))
                labels_batch.append(train_labels_encoded[idx])
            idx = (idx+1)%maxIdx
        yield np.array(train_batch), np.array(labels_batch)

def steps_per_epoch(train_files, train_labels_encoded, augments = 20, img_per_batch=100):
    return int(math.ceil(len(train_files) /img_per_batch))
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=2, padding='same', activation='relu', 
                                 kernel_initializer=tf.keras.initializers.Orthogonal(),
                                 input_shape=(IMG_HW, IMG_HW, 3)))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=2, padding='same', activation='relu', 
                                 kernel_initializer=tf.keras.initializers.Orthogonal()))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(N_CLASSES, activation='softmax'))
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(0.00005),
              metrics=['accuracy'])
EPOCHS = 30
AUGMENTS = 2
IMG_PER_BATCH = 50
model.fit_generator(train_generator(train_files, train_labels_encoded, AUGMENTS, IMG_PER_BATCH), epochs=EPOCHS, steps_per_epoch=steps_per_epoch(train_files, train_labels_encoded, AUGMENTS, IMG_PER_BATCH), verbose=2)
def test_generator(test_files):
    for f in test_files:
        img = resize(skimage.io.imread(f)/255.)
        if img.shape==(256,256,4):
            img = skimage.color.rgba2rgb(img)
        yield np.array([img])
    
predicted_probs = model.predict_generator(test_generator(test_files), steps=len(test_files))
predicted_classes = np.argmax(predicted_probs, axis=1)
out_df = pd.DataFrame({'file':[f.split('/')[3] for f in test_files], 
                       'species': labelEncoder.inverse_transform(predicted_classes)})
out_df.to_csv('submission.csv', index=False)
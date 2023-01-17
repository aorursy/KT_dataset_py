import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import cv2
import random
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import os
import math
import seaborn as sns
labels_all = os.listdir("../input/fruits-360_dataset/fruits-360/Training/")
N_CLASSES = len(labels_all)
label_encoder = LabelEncoder()
label_encoder.fit(labels_all)

files = glob.glob("../input/fruits-360_dataset/fruits-360/Training/*/*.jpg")
test_files = glob.glob("../input/fruits-360_dataset/fruits-360/Test/*/*.jpg")
labels = [i.split('/')[5] for i in files]
test_labels = [i.split('/')[5] for i in test_files]

files, labels = sklearn.utils.shuffle(files, labels)
labels_encoded = label_encoder.transform(labels)
test_labels_encoded = label_encoder.transform(test_labels)

df = pd.DataFrame({'label':labels_encoded, 'file':files})

IMG_HW = 64
IMG_SHAPE = (IMG_HW,IMG_HW,3)
pd.Series(labels_encoded).plot.hist(bins=N_CLASSES);
def apply_jitter(img, max_jitter=4):
    pts1 = np.array(np.random.uniform(-max_jitter, max_jitter, size=(4,2))+np.array([[0,0],[0,IMG_HW],[IMG_HW,0],[IMG_HW,IMG_HW]])).astype(np.float32)
    pts2 = np.array([[0,0],[0,IMG_HW],[IMG_HW,0],[IMG_HW,IMG_HW]]).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,M,(IMG_HW,IMG_HW))

def apply_rotation(img, angle):
    M = cv2.getRotationMatrix2D((IMG_HW/2,IMG_HW/2),angle,1)
    return cv2.warpAffine(img,M,(IMG_HW,IMG_HW))

def resize_keep_aspect(img, shape):
    h, w, _ = img.shape
    maxHW = max(h,w)
    top = (maxHW-h)//2
    bottom = (maxHW-h) - top
    left = (maxHW-w)//2
    right = (maxHW-w) - left
    withBorder = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_WRAP)
    return skimage.transform.resize(withBorder, shape)

def train_generator(files, labels, augments=5, imgs_per_batch=10) :
    while True:        
        for i in range(0, len(files), imgs_per_batch):
            img_batch = []
            label_batch = []
            for j in range(i, min(len(files), i+imgs_per_batch)):
                file = files[j]
                img = skimage.io.imread(file)/255.
                label = labels[j]
                img_resized = resize_keep_aspect(img, (IMG_SHAPE[0],IMG_SHAPE[1]))
                img_resized = skimage.color.rgb2yuv(img_resized)
                img_batch.append(img_resized)
                label_batch.append(label)
                for _ in range(augments):
                    img_batch.append(apply_rotation(apply_jitter(img_resized), random.randint(0,360)))
                    label_batch.append(label)
            yield np.array(img_batch), np.array(label_batch)

def steps_per_epoch(train_files, train_labels_encoded, augments=5, img_per_batch=10):
    return int(math.ceil(len(train_files) /img_per_batch))
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(5,5), strides=2, padding='same', activation='relu', 
                                 kernel_initializer=tf.keras.initializers.Orthogonal(),
                                 input_shape=(IMG_HW, IMG_HW, 3)))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(2014, activation='relu'))
model.add(tf.keras.layers.Dense(N_CLASSES, activation='softmax'))
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(0.0015),
              metrics=['accuracy'])
EPOCHS = 10
AUGMENTS = 3
IMG_PER_BATCH = 30
model.fit_generator(train_generator(files, labels_encoded, AUGMENTS, IMG_PER_BATCH), epochs=EPOCHS, steps_per_epoch=steps_per_epoch(files, labels_encoded, AUGMENTS, IMG_PER_BATCH), verbose=2)
def predict_generator(test_files):
    while True:        
        for i in range(0, len(test_files)):
            file = test_files[i]
            img = skimage.io.imread(file)/255.
            img_resized = resize_keep_aspect(img, (IMG_SHAPE[0],IMG_SHAPE[1]))
            img_resized = skimage.color.rgb2yuv(img_resized)
            yield np.array([img_resized])

predicted_probs = model.predict_generator(predict_generator(test_files), steps=len(test_files))
predicted_classes = np.argmax(predicted_probs, axis=1)
print(sklearn.metrics.accuracy_score(test_labels_encoded, predicted_classes))
sns.heatmap(sklearn.metrics.confusion_matrix(test_labels_encoded, predicted_classes));
model.save('model.h5')
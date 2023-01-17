import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import cv2
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import os
import math

!mkdir -p ~/.keras/models/
!cp ../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/
labels_all = os.listdir("../input/flowers-recognition/flowers/flowers")
N_CLASSES = len(labels_all)
label_encoder = LabelEncoder()
label_encoder.fit(labels_all)

imgs = glob.glob("../input/flowers-recognition/flowers/flowers/*/*.jpg")
labels = [i.split('/')[5] for i in imgs]

imgs, labels = sklearn.utils.shuffle(imgs, labels)
labels_encoded = label_encoder.transform(labels)

df = pd.DataFrame({'label':labels_encoded, 'file':imgs})

IMG_HW = 256
IMG_SHAPE = (IMG_HW,IMG_HW,3)
def apply_jitter(img, max_jitter=10):
    pts1 = np.array(np.random.uniform(-max_jitter, max_jitter, size=(4,2))+np.array([[0,0],[0,IMG_HW],[IMG_HW,0],[IMG_HW,IMG_HW]])).astype(np.float32)
    pts2 = np.array([[0,0],[0,IMG_HW],[IMG_HW,0],[IMG_HW,IMG_HW]]).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,M,(IMG_HW,IMG_HW))

def resize_keep_aspect(img, shape):
    h, w, _ = img.shape
    maxHW = max(h,w)
    top = (maxHW-h)//2
    bottom = (maxHW-h) - top
    left = (maxHW-w)//2
    right = (maxHW-w) - left
    withBorder = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_WRAP)
    return skimage.transform.resize(withBorder, shape)

def train_generator(files, labels, augments, imgs_per_batch=10) :
    while True:        
        for i in range(0, len(files), imgs_per_batch):
            img_batch = []
            label_batch = []
            for j in range(i, min(len(files), i+imgs_per_batch)):
                file = files[j]
                img = skimage.io.imread(df.loc[j, 'file'])/255.
                label = labels[j]
                img_resized = resize_keep_aspect(img, (IMG_SHAPE[0],IMG_SHAPE[1]))
                img_resized = skimage.color.rgb2yuv(img_resized)
                img_batch.append(img_resized)
                label_batch.append(label)
                for _ in range(augments):
                    img_batch.append(apply_jitter(img_resized))
                    label_batch.append(label)
            yield np.array(img_batch), np.array(label_batch)
            
img = skimage.io.imread(df.loc[0, 'file'])/255.
plt.imshow(skimage.transform.resize(img, (IMG_SHAPE[0],IMG_SHAPE[1])))
resnet = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=IMG_SHAPE
)

for layer in resnet.layers:
    layer.trainable = False

l = resnet.output
l = tf.keras.layers.Flatten()(l)
l = tf.keras.layers.BatchNormalization()(l)
l = tf.keras.layers.Dense(2048, activation="relu")(l)
l = tf.keras.layers.BatchNormalization()(l)
l = tf.keras.layers.Dense(1024, activation="relu")(l)
l = tf.keras.layers.BatchNormalization()(l)
out = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(l)

model = tf.keras.models.Model(inputs=resnet.input, outputs=out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

EPOCH = 10
AUGMENTS = 2
IMGS_PER_BATCH = 10
model.fit_generator(train_generator(df['file'], df['label'], AUGMENTS, IMGS_PER_BATCH), epochs=EPOCH, steps_per_epoch=int(math.ceil(len(df)/IMGS_PER_BATCH)), verbose=1)
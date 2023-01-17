import tensorflow as tf
import keras 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import cv2
import os
import sys
import random
input_size = 331
from keras.preprocessing.image import ImageDataGenerator
data_dir = '../input/neuron-cy5-images/neuron cy5 full/Neuron Cy5 Full'

data_gen = ImageDataGenerator(samplewise_center=True,
                              samplewise_std_normalization=True)
image_gen = data_gen.flow_from_directory(data_dir, 
                                         target_size=(input_size,input_size),
                                         color_mode='grayscale',
                                         class_mode='categorical',
                                         batch_size=1,
                                         shuffle=True)

classes = dict((v, k) for k, v in image_gen.class_indices.items())
num_classes = len(classes)
num_samples = len(image_gen)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import VGG19
from tensorflow.python.keras.layers import GlobalMaxPooling2D, Dense

pretrained_model = VGG19(include_top=False,
                         pooling='none',
                         input_shape=(input_size, input_size, 3),
                         weights=None)
x = GlobalMaxPooling2D()(pretrained_model.output)
x = Dense(2048, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)
vgg16_model = Model(pretrained_model.input, output)

cfg = vgg16_model.get_config()
cfg['layers'][0]['config']['batch_input_shape'] = (None, input_size, input_size, 1)
model = Model.from_config(cfg)

weights_dir = '../input/fitting-deeper-networks-vgg19/VGG19_weights.h5'
model.load_weights(weights_dir)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics= [])
y_true = np.empty([num_samples, 2])
y_pred = np.empty([num_samples, 2])
X = np.empty([input_size, input_size, 1])
for i in range(num_samples):
    prog = ' Progress: '+str(i+1)+'/'+str(num_samples)
    X, y_true[i,:] = next(image_gen)
    y_pred[i,:] = model.predict(X, steps=1)
    sys.stdout.write('\r'+prog)
sys.stdout.write('\rDone                ')
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true[:,0], y_pred[:,0].round())
df_cm = pd.DataFrame(cm, range(2), range(2))
sn.set(font_scale=1.4)
plt.figure(figsize=(15,15))
ax = sn.heatmap(df_cm, cmap='Blues', annot=True, cbar=False, annot_kws={"size": 16})# font size
ax.set_title('Confusion Matrix - without rotation')
plt.xticks([0.5, 1.5], classes.values(), fontsize=16)
plt.yticks([0.5,1.5], classes.values(), fontsize=16)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.show()
print(classification_report(np.vectorize(classes.get)(y_true[:,0]),np.vectorize(classes.get)(y_pred[:,0].round())))
weights_dir = '../input/vgg19-random-rotation/VGG19_weights.h5'
model.load_weights(weights_dir)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics= [])
y_true = np.empty([num_samples, 2])
y_pred = np.empty([num_samples, 2])
X = np.empty([input_size, input_size, 1])
for i in range(num_samples):
    prog = ' Progress: '+str(i+1)+'/'+str(num_samples)
    X, y_true[i,:] = next(image_gen)
    y_pred[i,:] = model.predict(X, steps=1)
    sys.stdout.write('\r'+prog)
sys.stdout.write('\rDone                ')
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true[:,0], y_pred[:,0].round())
df_cm = pd.DataFrame(cm, range(2), range(2))
sn.set(font_scale=1.4)
plt.figure(figsize=(15,15))
ax = sn.heatmap(df_cm, cmap='Blues', annot=True, cbar=False, annot_kws={"size": 16})# font size
ax.set_title('Confusion Matrix - with rotation')
plt.xticks([0.5, 1.5], classes.values(), fontsize=16)
plt.yticks([0.5,1.5], classes.values(), fontsize=16)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.show()
print(classification_report(np.vectorize(classes.get)(y_true[:,0]),np.vectorize(classes.get)(y_pred[:,0].round())))
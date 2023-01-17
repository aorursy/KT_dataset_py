import cv2

import numpy as np

from tensorflow.keras import layers

from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.callbacks import Callback, ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import metrics

import tensorflow as tf

from tqdm import tqdm
def build_model(pretrained):

    model = Sequential([

        pretrained,

        layers.GlobalAveragePooling2D(),

        layers.Dense(1, activation='sigmoid')

    ])

    

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(),

        metrics=['accuracy']

    )

    

    return model
base_path = '/kaggle/input/140k-real-and-fake-faces/real-vs-fake/'

image_gen = ImageDataGenerator(rescale=1./255.)



train_flow = image_gen.flow_from_directory(

    base_path + 'train/',

    target_size=(224, 224),

    batch_size=64,

    class_mode='binary'

)



valid_flow = image_gen.flow_from_directory(

    base_path + 'valid/',

    target_size=(224, 224),

    batch_size=64,

    class_mode='binary'

)



test_flow = image_gen.flow_from_directory(

    base_path + 'test/',

    target_size=(224, 224),

    batch_size=64,

    class_mode='binary'

)
densenet = DenseNet121(

    weights='/kaggle/input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    input_shape=(224,224,3)

)

model = build_model(densenet)

model.summary()
train_steps = 100000//64

valid_steps = 20000//64



model.fit_generator(

    train_flow,

    epochs=3,

    steps_per_epoch=train_steps,

    validation_data=valid_flow,

    validation_steps=valid_steps

)
y_pred = model.predict_generator(test_flow)

y_test = test_flow.classes
print("ROC AUC Score:", metrics.roc_auc_score(y_test, y_pred))

print("AP Score:", metrics.average_precision_score(y_test, y_pred))

print()

print(metrics.classification_report(y_test, y_pred > 0.5))
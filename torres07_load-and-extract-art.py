import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import load_model
os.listdir('../input/hands-on-art')
data_dir = '../input/art-movements/dataset/dataset/'
RESOLUTION = 150
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

train_generator = train_datagen.flow_from_directory(
        data_dir + 'train/',
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical', subset="training")

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

val_generator = val_datagen.flow_from_directory(
        data_dir + 'train/',
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical', subset="validation")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        data_dir + 'test/',
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
# Class labels
train_generator.class_indices
model = load_model('../input/hands-on-art/inception_v3_art.h5')
model.summary()
model.pop()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])

model.summary()
Y_pred = model.predict_generator(test_generator, steps=len(test_generator))
Y_pred.shape
Y_pred[0]

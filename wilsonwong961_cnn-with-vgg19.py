import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\

import os

import tensorflow as tf

from tensorflow.keras import layers, models

import matplotlib.pyplot as plt

import cv2

from pathlib import Path

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

print(os.listdir("../input/chest_xray/chest_xray"))
# Load and augment training data

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale=1/255, 

    rotation_range=20, 

    width_shift_range=0.2, 

    height_shift_range=0.2, 

    horizontal_flip=True

)



train_generator = train_datagen.flow_from_directory(

    '../input/chest_xray/chest_xray/train',

    batch_size=16,

    target_size=(128, 128),

    shuffle=True,

    class_mode='categorical'

)
# Load validation and test data

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

val_generator = val_datagen.flow_from_directory(

    '../input/chest_xray/chest_xray/val',

    batch_size=16,

    target_size=(128, 128),

    class_mode='categorical'

)



test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

test_generator = test_datagen.flow_from_directory(

    '../input/chest_xray/chest_xray/test',

    batch_size=16,

    target_size=(128, 128),

    class_mode='categorical'

)
# Use MobileNetV2 as base model for transfer learning

base_model = tf.keras.applications.VGG19(

    weights='imagenet',

    include_top=False,

    input_shape=(128, 128, 3)

)

base_model.trainable = False
# Define model

def model():

    model = base_model.output

    model = tf.keras.layers.GlobalAveragePooling2D()(model)

    model = tf.keras.layers.Dense(units=512, activation='relu')(model)

    model = tf.keras.layers.Dropout(0.7)(model)

    predictions = tf.keras.layers.Dense(units=2, activation='softmax')(model)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    return model
model = model()

model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.fit_generator(

    train_generator,

    epochs=20,

    verbose=1,

    validation_data=val_generator,

)
loss, test_acc = model.evaluate_generator(

    test_generator,

    steps=None,

    max_queue_size=10,

    workers=1,

    use_multiprocessing=True,

    verbose=1

)



print(test_acc)
model.save('model.h5')
# Load model for testing

model = tf.keras.models.load_model('model.h5')



test_dir = Path('../input/chest_xray/chest_xray/test')



normal_cases_dir = test_dir / 'NORMAL'

pneumonia_cases_dir = test_dir / 'PNEUMONIA'



normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')



test_data = []

test_labels = []
for img in normal_cases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (128, 128))

    if img.shape[2] == 1:

        img = np.dstack([img, img, img])

    else:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = tf.keras.utils.to_categorical(0, num_classes=2)

    test_data.append(img)

    test_labels.append(label)



for img in pneumonia_cases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (128, 128))

    if img.shape[2] == 1:

        img = np.dstack([img, img, img])

    else:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = tf.keras.utils.to_categorical(1, num_classes=2)

    test_data.append(img)

    test_labels.append(label)



test_data = np.array(test_data)

test_labels = np.array(test_labels)



print("Total number of test examples: ", test_data.shape)

print("Total number of labels:", test_labels.shape)
# Get predictions

preds = model.predict(test_data, batch_size=16)

preds = np.argmax(preds, axis=-1)



# Original labels

orig_test_labels = np.argmax(test_labels, axis=-1)



cm = confusion_matrix(orig_test_labels, preds)

plt.figure()

plot_confusion_matrix(cm, figsize=(12, 8), hide_ticks=True,

                      cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()



tn, fp, fn, tp = cm.ravel()



precision = tp/(tp+fp)

recall = tp/(tp+fn)



print("Recall of the model is {:.2f}".format(recall))

print("Precision of the model is {:.2f}".format(precision))
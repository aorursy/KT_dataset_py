# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames[:2]:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install --user opencv-python
import cv2

import numpy as np

from tensorflow.keras import layers

from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.callbacks import Callback, ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import metrics

import tensorflow as tf
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
"""

Plot the training and validation loss

epochs - list of epoch numbers

loss - training loss for each epoch

val_loss - validation loss for each epoch

"""

def plot_loss(epochs, loss, val_loss):

    plt.plot(epochs, loss, 'bo', label='Training Loss')

    plt.plot(epochs, val_loss, 'orange', label = 'Validation Loss')

    plt.title('Training and Validation Loss')

    plt.legend()

    plt.show()

    

    

"""

Plot the training and validation accuracy

epochs - list of epoch numbers

acc - training accuracy for each epoch

val_acc - validation accuracy for each epoch

"""

def plot_accuracy(epochs, acc, val_acc):

    plt.plot(epochs, acc, 'bo', label='Training accuracy')

    plt.plot(epochs, val_acc, 'orange', label = 'Validation accuracy')

    plt.title('Training and Validation Accuracy')

    plt.legend()

    plt.show()
# base_path = '../combined-real-and-fake-faces/combined-real-vs-fake/'

base_path = '/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/'

image_gen = ImageDataGenerator()



train_flow = image_gen.flow_from_directory(

    base_path + 'train/',

    target_size=(224, 224),

    batch_size=64,

    color_mode='grayscale',

    class_mode='binary'

)



valid_flow = image_gen.flow_from_directory(

    base_path + 'valid/',

    target_size=(224, 224),

    batch_size=64,

    color_mode='grayscale',

    class_mode='binary'

)

test_flow = image_gen.flow_from_directory(

    base_path + 'test/',

    target_size=(224, 224),

    batch_size=1,

    color_mode='grayscale',

    shuffle = False,

    class_mode='binary'

)
densenet = DenseNet121(

    weights=None,

    include_top=False,

    input_shape=(224,224,1)

)

model = build_model(densenet)

model.summary()
train_steps = 100000//64

valid_steps = 20000//64



history = model.fit_generator(

    train_flow,

    epochs = 10,

    steps_per_epoch = train_steps,

    validation_data = valid_flow,

    validation_steps = valid_steps 

)
model.save('grayscale_densenet.h5')
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']
plot_loss(range(1, len(loss) + 1), loss, val_loss)
plot_accuracy(range(1, len(loss) + 1), acc, val_acc)
y_pred = model.predict(test_flow)

y_test = test_flow.classes
print("ROC-AUC Score:", metrics.roc_auc_score(y_test, y_pred))

print("AP Score:", metrics.average_precision_score(y_test, y_pred))

print()

print(metrics.classification_report(y_test, y_pred > 0.5))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import random

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout

import cv2

import numpy as np

import random
CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space']

DATADIR = "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"

training_data = []
seed = 0

np.random.seed(seed)

random.seed(seed)
for category in CATEGORIES:

    path = os.path.join(DATADIR, category)

    class_num = CATEGORIES.index(category)

    print('Category: ', category)

    count = 0

    for img in os.listdir(path):

        # if count == 1000: break

        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

        new_array = cv2.resize(img_array, (64, 64))

        training_data.append([new_array, class_num])

        count += 1

        if count % 100 == 0:

            print(count, end=' ')

    print()
random.shuffle(training_data)



X = []

y = []



for features, label in training_data:

    X.append(features)

    y.append(label)



training_data = []
X = np.array(X).reshape(-1, 64, 64, 1)

# X = np.array(X).reshape(X.shape[0], 28, 28, 1)

y = np.array(y).astype(float)
X = X / 255.0
model = Sequential()

model.add(Conv2D(512, (4, 4), input_shape=X.shape[1:]))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(3, 3)))



model.add(Conv2D(256, (3, 3)))

model.add(Activation("relu"))

# model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation("relu"))

model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))



model.add(Flatten())

# model.add(Dense(64))

# model.add(Activation("relu"))



model.add(Dense(len(CATEGORIES)))

model.add(Activation('softmax'))
model.compile(

        optimizer=tf.optimizers.Adam(learning_rate=0.001),

        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="auto", name="sparse_categorical_crossentropy"),

        metrics=['accuracy']        

        )

model.fit(X, y, epochs=10, validation_split=0.1)
result = []

for category in CATEGORIES:

    image = cv2.imread('/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/'+str(category)+'_test.jpg', 0)

    image = cv2.resize(image, (64, 64))

    image = image.reshape(-1, 64, 64, 1).astype(float)

    prediction = model.predict_classes(image)

    result.append([category, CATEGORIES[prediction[0]]])
count = 0

for i, j in result:

    if i == j:

        count += 1

print('Correct Predictioins:', count)

print('Incorrect Prediction:', len(CATEGORIES) - count)

print('Test Accuracy:', (count / len(CATEGORIES) * 100))
# print(result)

for exp, pred in result:

    print(f'Expected: {exp} -> Predicted: {pred}') 
tf.keras.models.save_model(

    model,

    './image_model_new.h5',

    overwrite=True,

    include_optimizer=True

)
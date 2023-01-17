# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
infected = '../input/cell-images-for-detecting-malaria/cell_images/Parasitized'
uninfected = '../input/cell-images-for-detecting-malaria/cell_images/Uninfected'
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from tensorflow.keras.layers import Conv2D,MaxPooling2D
# from tensorflow.keras.applications.vgg19 import VGG19
infected_images = []
for file in os.listdir(infected):
    if file == 'Thumbs.db':
        pass
    else:
        img = Image.open(os.path.join(infected, file))
        img = img.resize((36,36))
        img = np.asarray(img)
        infected_images.append(img)
uninfected_images = []
for file in os.listdir(uninfected):
    if file == 'Thumbs.db':
        pass
    else:
        img = Image.open(os.path.join(uninfected, file))
        img = img.resize((36,36))
        img = np.asarray(img)
        uninfected_images.append(img)
for i in range(5):
    plt.imshow(infected_images[i])
    plt.show()
for i in range(5):
    plt.imshow(uninfected_images[i])
    plt.show()
images = np.asarray(infected_images + uninfected_images)
images.shape
labels = np.asarray([1 for _ in range(len(infected_images))] + [0 for _ in range(len(uninfected_images))])
labels.shape
from sklearn.utils import shuffle
images, labels = shuffle(images, labels)
for i in range(10):
    print(labels[i])
    plt.imshow(images[i])
    plt.show()
# np.save('images.npy', images)
# np.save('labels.npy', labels)
from tensorflow.keras.utils import to_categorical
labels = to_categorical(labels, num_classes = 2)
labels[:10]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 42)
print("Train size:", X_train.shape, y_train.shape)
print("Test size:", X_test.shape, y_test.shape)
model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape = X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(512, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(96, activation='relu'))

model.add(Dense(2, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), verbose = 1)
pred = [np.argmax(i) for i in model.predict(X_test)]
pred[:5]
tru = [np.argmax(i) for i in y_test]
from sklearn.metrics import confusion_matrix
confusion_matrix(tru, pred)
model.save('malaria1.h5')
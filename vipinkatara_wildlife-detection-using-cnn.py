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
        break

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATA_DIR = '/kaggle/input/african-wildlife'
CATEGORIES = os.listdir(DATA_DIR)
IMG_SIZE=100
CATEGORIES

data = []
for category in CATEGORIES:
    label = CATEGORIES.index(category)
    path = os.path.join(DATA_DIR, category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img))
            new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            data.append([label,new_array])
        except Exception as E:
            pass
data
#checking shape
np.shape(data)
X = []
Y = []

for label,feature in data:
    X.append(feature)
    Y.append(label)


#checking shape of features
print(np.shape(X))
#checking shape of labels
print(np.shape(Y))
#converting both array to numpy array
X = np.array(X)
Y = np.array(Y)
X = X / 255
X
np.max(X)
Y
val = np.random.randint(0, len(X), 12)
r = 1
plt.figure(figsize=(25,20))
for value in val:
    plt.subplot(3,4,r)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    plt.xticks([]) , plt.yticks([])
    plt.title(CATEGORIES[Y[value]])
    r += 1
    plt.imshow(X[value])
plt.show()
Y = to_categorical(Y)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.15,random_state=0)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=0)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
model = Sequential()
model.add(Conv2D(64,3,padding='same', input_shape=(100,100,3)))
model.add(MaxPool2D((4,4)))
model.add(Conv2D(32,3,padding='same', input_shape=(100,100,3)))
model.add(MaxPool2D((4,4)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(4))

model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range = 10,
    zoom_range = 0.1,
    width_shift_range = 0.1,
    height_shift_range = 0.1
)
datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=32), epochs=100, validation_data=(x_val,y_val))
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred.round()))
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input"))
data = pd.read_csv('/kaggle/input/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv')
data.shape
data.head()
data.Usage.value_counts()
emotion_map = {0: ' Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = data['emotion'].value_counts(sort=False).reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)
emotion_counts
plt.figure(figsize=(6, 4))
plt.bar(emotion_counts.emotion, emotion_counts.number)
plt.title('Class distribution')
plt.xlabel('Emotions', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.show()
x = []
y = []
first = True
for line in open("/kaggle/input/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv"):
    if first:
        first = False
    else:
        row = line.split(',')
        x.append([int(p) for p in row[1].split()])
        y.append(int(row[0]))
x, y = np.array(x) / 255.0, np.array(y)
x = x.reshape(-1, 48, 48, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
num_class = len(set(y))
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)
datagen = ImageDataGenerator(
    zoom_range = 0.2,
    height_shift_range = 0.1,
    width_shift_range = 0.1,
    rotation_range = 10,
    horizontal_flip = True,
    vertical_flip = True
)
datagen.fit(x_train)
model = Sequential()

input_shape = (48, 48, 1)

model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=128, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(filters=512, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(7, activation='softmax'))

optimizer = RMSprop(learning_rate=0.001, rho=0.99)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, verbose=1,patience=2, min_lr=0.00000001)

callback = EarlyStopping(monitor='loss', patience=5)

model.summary()
history = model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=20, validation_data=(x_test, y_test), verbose=1, callbacks=[reduce_lr, callback])
final_loss, final_acc = model.evaluate(x_test, y_test, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4}".format(final_loss, final_acc))
plt.plot(history.history['loss'], color='b')
plt.plot(history.history['val_loss'], color='r')
plt.show()
plt.plot(history.history['accuracy'], color='b')
plt.plot(history.history['val_accuracy'], color='r')
plt.show()
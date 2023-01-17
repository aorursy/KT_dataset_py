import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from glob import glob
import cv2
import seaborn as sns
import os
labels = ['PNEUMONIA', 'NORMAL']
img_size = 150
def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)
            
train = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/train')
val = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/val')
test = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
l = []
for i in train:
    if i[1] == 0:
        l.append("PNEUMONIA")
    else:
        l.append("NORMAL")
sns.set_style('darkgrid')
sns.countplot(l)
plt.figure(figsize=(5, 5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])

plt.figure(figsize=(5, 5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])
x = []
y = []

for feature, label in train:
    x.append(feature)
    y.append(label)
    
for feature, label in test:
    x.append(feature)
    y.append(label)
    
for feature, label in val:
    x.append(feature)
    y.append(label)
x = np.array(x).reshape(-1, img_size, img_size, 1)
y = np.array(y)

x_train, x_further, y_train, y_further = train_test_split(x, y, test_size=0.2, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_further, y_further, test_size=0.5)

x_train = x_train/255
x_test = x_test/255
x_val = x_val/255
datagen = ImageDataGenerator(
    rotation_range = 30,
    zoom_range = 0.1,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True
)

datagen.fit(x_train)
model = Sequential()

input_shape=(150, 150, 1)

model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=2, padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D((3, 3), strides=2, padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()
optimizer = Adam(lr = 0.0001)
early_stopping_monitor = EarlyStopping(patience=3, monitor="val_accuracy", restore_best_weights=True)
model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=optimizer)
history = model.fit(datagen.flow(x_train,y_train, batch_size=32), epochs = 15, validation_data=datagen.flow(x_val, y_val), callbacks=[early_stopping_monitor])
final_loss, final_acc = model.evaluate(x_test, y_test)
print("Final loss: {0:.4f}, final accuracy: {1:.4}".format(final_loss, final_acc))
plt.plot(history.history['accuracy'], color='b')
plt.plot(history.history['val_accuracy'], color='r')
plt.show()

plt.plot(history.history['loss'], color='b')
plt.plot(history.history['val_loss'], color='r')
plt.show()


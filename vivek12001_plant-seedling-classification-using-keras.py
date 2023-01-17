import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
from IPython.display import clear_output
categories = os.listdir("/kaggle/input/plant-seedlings-classification/train")
categories
x = []
y = []
width, height = 140, 140

count = 0
path = "/kaggle/input/plant-seedlings-classification/"
types = ["train"]

for t in types:
    datadir = os.path.join(path, t)
    for c in categories:
        path = os.path.join(datadir, c)
        class_no = categories.index(c)
        for img in os.listdir(path):
            image = os.path.join(path, img)
            image = cv2.imread(image, cv2.IMREAD_ANYCOLOR)
            image = cv2.resize(image, (width, height))
            x.append(image)
            x.append(cv2.flip(image, -1))
            y.append(class_no)
            y.append(class_no)
            count += 1
    clear_output()
    print(count)
x = np.array(x)/255
y = np.array(y)
x.shape
def one_hot_encoding(labels, c):
    one_hot_matrix = tf.one_hot(labels, c)
    return tf.keras.backend.eval(one_hot_matrix)
y = one_hot_encoding(y, 12)

print(str(y.shape))
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=8)
del x
del y
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential()

input_shape = (140, 140, 3)

model.add(Conv2D(filters=64, kernel_size=1, padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=2, padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D((3, 3), strides=2, padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(50, activation='relu'))

model.add(Dense(12, activation='softmax'))

model.summary()
initial_learning_rate = 0.001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=800,
    decay_rate=0.5,
    staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
result = model.fit(x=x_train,y=y_train,batch_size=64,epochs=80,verbose=1,shuffle=False,initial_epoch=0,
                   validation_split=0.2)
final_loss, final_acc = model.evaluate(x_val, y_val)
print("Final loss: {0:.4f}, final accuracy: {1:.4}".format(final_loss, final_acc))
plt.plot(result.history['accuracy'], label='train')
plt.plot(result.history['val_accuracy'], label='valid')
plt.legend(loc='upper left')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
plt.plot(result.history['loss'], label='train')
plt.plot(result.history['val_loss'], label='test')
plt.legend(loc='upper right')
plt.title('Model Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()
X = []
file = []
def createTestData():
    a=0
    path = "/kaggle/input/plant-seedlings-classification/"
    types = ["test"]
    for t in types:
        PATH = os.path.join(path,t)
        for img in os.listdir(PATH):
            file.append(img)
            image = os.path.join(PATH, img)
            image = cv2.imread(image, cv2.IMREAD_ANYCOLOR)
            image = cv2.resize(image , (width, height))
            X.append(image)
            a+=1
    print(a)
createTestData()
X = np.array(X)/255
X.shape
species = model.predict_classes(X)
print(species.shape)
print(species)
ans = pd.DataFrame(file,columns = ["file"])
ans = ans.join(pd.DataFrame(species,columns=["species"]))
ans["species"] = ans["species"].apply(lambda x: categories[int(x)])
ans.head(794)
ans.to_csv("answers.csv",index=False)
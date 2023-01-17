import pandas as pd

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
print(df_train.shape)
print(df_test.shape)
y_train = df_train.iloc[:,0]
x_train = df_train.iloc[:,1:]
x_test = df_test

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
y_train = y_train.values
x_train = x_train.values
x_test = x_test.values

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
x_train = x_train.reshape(42000, 28, 28)
x_test = x_test.reshape(28000, 28, 28)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
#---------------------------------
import matplotlib.pyplot as plt
import numpy as np

j = 331
for i in range(9):
    plt.subplot(331+i)
    random_num = np.random.randint(0,len(x_train))
    plt.axis('off')
    plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print(x_train.shape)
print(x_test.shape)
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)

print(y_train.shape)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
num_classes = 10

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
from keras.optimizers import SGD

model.compile(loss='categorical_crossentropy', optimizer=SGD(0.01), metrics=['accuracy'])
print(model.summary())
batch_size = 32
epochs = 20

history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
                    verbose=1)

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, loss_values, label='Training Loss', color='r')
plt.setp(line1, linewidth=2, marker='4', markersize=10)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()

acc_values = history_dict['accuracy']
line2 = plt.plot(epochs, acc_values, label='Training Accuracy')
plt.setp(line2, linewidth=2, marker='x', markersize=10)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
#----------------Predicting Test Data------------------

j = 331
for i in range(6):
    plt.figure(figsize=(2,2))
    random_num = np.random.randint(0,len(x_test))
    num = x_test[random_num].reshape(1,28,28,1)
    plt.title(model.predict_classes(num))
    num = num.reshape(28,28)
    plt.axis('off')
    plt.imshow(num, cmap=plt.get_cmap('gray'))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# %load_ext tensorboard

# %tensorboard --logdir logs

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input, ZeroPadding2D,GlobalAveragePooling2D,MaxPool2D

from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.utils import to_categorical

from keras.utils import plot_model

import matplotlib.image as mpimg

import os

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#pip show tensorflow
from sklearn.model_selection import train_test_split



#load the data 

x = np.load('/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy')

y = np.load('/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy')



print("x - max :",x.max())

print("x - min :",x.min())

print("x - shape :",x.shape)



print("y - max :",y.max())

print("y - min :",y.min())

print("y - shape :",y.shape)

import matplotlib.pyplot as plt

import matplotlib.image as img

def show_sample():

  plt.figure(figsize=(10,10))

  for n in range(25):

      ax = plt.subplot(5,5,n+1)

      plt.imshow(x[n])

      plt.axis('off')

show_sample()
# Now,lets create x_train, y_train, x_test, y_test arrays

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

#reshape

x_train = x_train.reshape(-1,64,64,1)

x_test = x_test.reshape(-1,64,64,1)

#print x_train and y_train shape

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
def buildModel():

    model = Sequential()

    model.add(Conv2D(32,(3,3), padding='same', input_shape=(64, 64,1)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    # 2nd Convolution layer

    model.add(Conv2D(64,(1,1), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    # 3rd Convolution layer

    model.add(Conv2D(128,(3,3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    # 4th Convolution layer

    model.add(Conv2D(256,(3,3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    # Flattening

    model.add(Flatten())



    # Fully connected layer 1st layer

    model.add(Dense(512))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.25))



    model.add(Dense(10, activation='softmax'))

    return model
model = buildModel().summary()
model = buildModel()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard , CSVLogger, ReduceLROnPlateau

import datetime

checkpoint = ModelCheckpoint("model_vgg.h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3,verbose=1,mode='auto', min_delta=0.0001)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



csvlogger = CSVLogger(filename= "training_csv.log", separator = ",", append = False)

callbacks = [checkpoint,early]

#batch size = 2**x , 16,32,64,24 

epoch = 100

BATCH_SIZE = 32

learning_rate = 0.0001
from keras.optimizers import Adam

model_adam = buildModel()

optimizer = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999)

model_adam.compile(optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

history = model_adam.fit(x_train, y_train,epochs=epoch, batch_size=BATCH_SIZE,validation_data=(x_test,y_test),callbacks=callbacks)

scores = model_adam.evaluate(x_test, y_test, verbose=0)

loss_valid=scores[0]

acc_valid=scores[1]



print('-------------------ADAM-----------------------------------------')

print("validation loss: {:.2f}, validation accuracy: {:.01%}".format(loss_valid, acc_valid))

print('---------------------------------------------------------------')
acc = history.history['accuracy']

loss = history.history['loss']



val_acc = history.history['val_accuracy']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.figure(figsize = (16, 5))



plt.subplot(1,2,1)

plt.plot(epochs, acc, 'r', label = 'Training Accuracy')

plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')

plt.title('Training vs. Validation Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()



plt.subplot(1,2,2)

plt.plot(epochs, loss, 'r', label = 'Training Loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')

plt.title('Training vs. Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()



plt.show()
from sklearn.metrics import confusion_matrix

import seaborn as sns

y_head = model_adam.predict(x_test)



confusion_matrix= confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_head, axis=1))

class_names=[0,1,2,3,4,5,6,7,8,9]



fig, ax = plt.subplots(figsize=(10,10))

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="Purples" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion Matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
from keras import models

load_model = models.load_model("model_vgg.h5")

load_model.layers[0].input_shape

img = x[800]

test_img = img.reshape((-1,64,64,1))

img_class = load_model.predict_classes(test_img)

classname = img_class[0]

print("Class: ",classname)

plt.imshow(img)

plt.title(classname)

plt.show()
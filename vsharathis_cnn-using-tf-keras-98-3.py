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
train_df=pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')

test_df=pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')
train_df.head()
train_df.info()
train_df.describe()
train_label=train_df['label']

train_label.head()

trainset=train_df.drop(['label'],axis=1)

trainset.head()
X_train = trainset.values

X_train = trainset.values.reshape(-1,28,28,1)

print(X_train.shape)
test_label=test_df['label']

X_test=test_df.drop(['label'],axis=1)

#print(X_test.head())

X_test = X_test.values.reshape(-1,28,28,1)

print(X_test.shape)
from sklearn.preprocessing import LabelBinarizer

lb=LabelBinarizer()

y_train=lb.fit_transform(train_label)

y_test=lb.fit_transform(test_label)

y_train
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255,fill_mode='nearest')

datagen.fit(X_train)

X_test=X_test/255
import matplotlib.pyplot as plt

import matplotlib.image as img

def show_sample():

  plt.figure(figsize=(10,10))

  for n in range(25):

      ax = plt.subplot(5,5,n+1)

      plt.imshow(X_train[n].reshape(28,28))

      plt.axis('off')

show_sample()
import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input, ZeroPadding2D,GlobalAveragePooling2D,MaxPool2D,Activation

def buildModel():

    model = tf.keras.Sequential()

    model.add(Conv2D(32,(3,3), padding='same', input_shape=(28, 28,1)))

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

    

    model.add(Dense(512))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(Dense(24, activation='softmax'))

    return model
model = buildModel().summary()
model = buildModel()

tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard , CSVLogger, ReduceLROnPlateau

checkpoint = ModelCheckpoint("model_sign.h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

callbacks = [checkpoint,early]
#batch size = 2**x , 16,32,64,24 

epoch = 100

BATCH_SIZE = 32

learning_rate = 0.0001
from tensorflow.keras.optimizers import Adam

model_adam = buildModel()

optimizer = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999)

model_adam.compile(optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

history = model_adam.fit_generator(datagen.flow(X_train,y_train,batch_size=BATCH_SIZE),epochs=epoch,validation_data=(X_test,y_test),callbacks=callbacks)

scores = model_adam.evaluate(X_test, y_test, verbose=0)

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
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

import seaborn as sns

predictions = model_adam.predict(X_test)



confusion_matrix= confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))

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
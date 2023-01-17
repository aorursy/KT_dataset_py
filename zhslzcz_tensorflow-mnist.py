import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
train.head()
train.shape
X_train = (train.iloc[:,1:].values)

y_train = (train.iloc[:,0].values)
X_train = X_train.reshape(X_train.shape[0], 28, 28)
for i in range(0, 9):

    plt.subplot(330+(i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.axis('off')

    plt.title(y_train[i]);
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_train.shape
def Normalize(X): 

    X=X / 255.0

    mean_px = X.mean()

    std_px = X.std()

    return (X-mean_px)/std_px.astype(np.float32)
X_train = Normalize(X_train)
# from keras.utils.np_utils import to_categorical

# y_train= to_categorical(y_train)

# num_classes = y_train.shape[1]

# num_classes
def history_plot(history):

    plt.figure(figsize=[15,7])

    plt.subplot(1,2,1)

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    

    plt.subplot(1,2,2)

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
import tensorflow as tf
annmodel = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28,1)),

    tf.keras.layers.Dense(512, activation=tf.nn.relu),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

annmodel.compile(optimizer='adam',

              #loss='categorical_crossentropy',#you need to convert to one-hot label

              loss='sparse_categorical_crossentropy', # no need of one-hot label

              metrics=['accuracy'])



history = annmodel.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

history_plot(history)
cnnmodel = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),

  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(10, activation='softmax')

])

cnnmodel.summary()
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('acc')>0.995):

          print("\nReached 99.5% accuracy so cancelling training!")

          self.model.stop_training = True



callbacks = myCallback()
cnnmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history =cnnmodel.fit(X_train, y_train, batch_size=32, epochs=10,validation_split=0.2,callbacks=[callbacks])

history_plot(history)
from tensorflow.keras import models



def convLayerPlot(model,layer_number,CONV_number,image_source,image_numbers):

    f, axarr = plt.subplots(len(image_numbers),layer_number)

    layer_outputs = [layer.output for layer in model.layers]

    activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)



    for x in range(0,layer_number):

        for i,n in enumerate(image_numbers):

            f1 = activation_model.predict(image_source[n].reshape(1, 28, 28, 1))[x]

            axarr[i,x].imshow(f1[0, : , :, CONV_number], cmap='inferno')

            axarr[i,x].grid(False)
convLayerPlot(cnnmodel,2,0,X_train,[1,2,8])
testpic=np.array(X_train[7])

plt.imshow(testpic.reshape(28, 28), cmap=plt.get_cmap('gray'))
testpic[0:4,0:6,0]=2

plt.imshow(testpic.reshape(28, 28), cmap=plt.get_cmap('gray'))
cnnmodel.predict(testpic.reshape(1,28,28,1))[0][3]
testpic=np.array(X_train[7])

testpic[10:15,5:10,0]=2

plt.imshow(testpic.reshape(28, 28), cmap=plt.get_cmap('gray'))
cnnmodel.predict(testpic.reshape(1,28,28,1))[0][3]
import random
def affectingarea(testpic,maxsize,model,alpha=0.3):

    r,c=len(testpic),len(testpic[0])

    pred=model.predict(testpic.reshape(1,r,c,1))

    predclass=np.argmax(pred)

    origpred=pred[0][predclass]

    minv,maxv = np.amin(testpic),np.amax(testpic)

    #print(minv,maxv)

    importance=np.zeros((r,c))

    for size in range(1,maxsize):

        for i in range(0,r+1-size):

            for j in range(0,r+1-size):

                #area=[i+x,j+y for x in range(size) for y in range(size)]

                temppic=np.array(testpic)

                temppic[i:i+size,j:j+size,0] = random.uniform(minv,maxv)

                predict=model.predict(temppic.reshape(1,r,c,1))[0][predclass]

                importance[i:i+size,j:j+size]+=(1-predict)/(size**2)  #normalized by num of pixels 

    plt.imshow(importance,cmap=plt.get_cmap('Reds'))

    plt.imshow(testpic.reshape(r,c), cmap=plt.get_cmap('gray'),alpha=alpha)

    return importance
numberlist=[1,2,16,7,3,8,21,18,10,11]

plt.figure(figsize=(12,6))

for i,n in enumerate(numberlist):

    plt.subplot(2,5,i+1)

    imp=affectingarea(X_train[n],10,cnnmodel)

    #plt.imshow(X_train[i].reshape(28, 28), cmap=plt.get_cmap('gray'))

    plt.axis('off')

    plt.title(y_train[n]);
plt.figure(figsize=(12,6))

for i,n in enumerate(numberlist):

    plt.subplot(2,5,i+1)

    imp=affectingarea(X_train[n],10,annmodel)

    #plt.imshow(X_train[i].reshape(28, 28), cmap=plt.get_cmap('gray'))

    plt.axis('off')

    plt.title(y_train[n]);
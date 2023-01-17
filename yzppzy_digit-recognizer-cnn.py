import tensorflow as tf

from tensorflow import keras

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
tf.__version__
data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
img_data=data.iloc[:,1:]

label_data=data.iloc[:,0]
train_img=img_data.iloc[0:30000,:]

test_img=img_data.iloc[30000:,:]

train_img.head()
train_label=label_data.iloc[0:30000]

test_label=label_data.iloc[30000:]

train_label.head()
train_img.shape
test_img.shape
train_img=np.array(train_img).reshape((30000,28,28))

test_img=np.array(test_img).reshape((12000,28,28))
for i in range(0,25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_img[i],cmap='binary')
train_img=train_img/255

test_img=test_img/255
train_img=np.expand_dims(train_img,-1)

test_img=np.expand_dims(test_img,-1)
model=keras.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu',padding='same'))

model.add(keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'))

model.add(keras.layers.MaxPool2D())

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'))

model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'))

model.add(keras.layers.MaxPool2D())

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'))

model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'))

model.add(keras.layers.GlobalAveragePooling2D())

model.add(keras.layers.Dense(256,activation='relu'))

model.add(keras.layers.Dense(10,activation='softmax'))
model.compile(keras.optimizers.Adam(learning_rate=0.001),

             loss='sparse_categorical_crossentropy',

             metrics=['acc'])
history=model.fit(train_img,train_label,epochs=20,validation_data=(test_img,test_label))
plt.plot(history.epoch,history.history.get('acc'),label='acc')

plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')

plt.legend()
plt.plot(history.epoch,history.history.get('loss'),label='loss')

plt.plot(history.epoch,history.history.get('val_loss'),label='val_loss')

plt.legend()
test=np.array(test).reshape((28000,28,28))
test=np.expand_dims(test,-1)
test.shape
results=model.predict(test)
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
results
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission
submission.to_csv("cnn_mnist_datagen.csv",index=False)
import pandas as pd
import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPool2D
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train=pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
test=pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")
train.info()
test.info()
train.describe()
train.head()
train_label=train["label"]
train_label.head()
trainset=train.drop(["label"],axis=1)
trainset.head()
x_train=trainset.values
x_train=trainset.values.reshape(-1,28,28,1)
print(x_train.shape)
test_label=test["label"]
x_test=test.drop(["label"],axis=1)
print(x_test.shape)
x_test.head()
from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y_train=lb.fit_transform(train_label)
y_test=lb.fit_transform(test_label)
y_train
x_test=x_test.values.reshape(-1,28,28,1)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
datagen=ImageDataGenerator(rescale=1./255,
                           rotation_range=0,
                           height_shift_range=0.2,
                          width_shift_range=0.2,
                           shear_range=0,zoom_range=0,
                           horizontal_flip=True,
                                  fill_mode='nearest')
x_test=x_test/255
fig,axe=plt.subplots(2,2)
fig.suptitle('Preview of dataset')
axe[0,0].imshow(x_train[0].reshape(28,28),cmap='gray')
axe[0,0].set_title('label: 3  letter: C')
axe[0,1].imshow(x_train[1].reshape(28,28),cmap='gray')
axe[0,1].set_title('label: 6  letter: F')
axe[1,0].imshow(x_train[2].reshape(28,28),cmap='gray')
axe[1,0].set_title('label: 2  letter: B')
axe[1,1].imshow(x_train[4].reshape(28,28),cmap='gray')
axe[1,1].set_title('label: 13  letter: M')
plt.figure(figsize=(10,5))
sns.countplot(train_label)
plt.title('Frequency of each label')
model=Sequential()
model.add(Conv2D(128,kernel_size=(5,5),strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(Conv2D(32,kernel_size=(2,2),strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(Flatten())
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24,activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(datagen.flow(x_train,y_train,batch_size=200),
         epochs = 20,
          validation_data=(x_test,y_test),
          shuffle=1
         )
acc=model.evaluate(x=x_test,y=y_test)

"model acc={}%".format(acc*100)
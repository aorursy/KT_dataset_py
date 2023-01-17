import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPool2D
from keras.activations import relu
import pandas as pd

classes=10
epochs=10
batchsize=128

df_train = pd.read_csv('../input/train.csv')
df_train.head()
df_test = pd.read_csv('../input/test.csv')
df_test.head()
from keras import backend as K
import numpy as np
K.set_image_data_format('channels_last')
np.random.seed(0)
y = df_train["label"]
df_train.drop(["label"], inplace = True, axis = 1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_train.values, y.values, test_size=0.2 , random_state=42)
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train.shape
x_test.shape
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
input_shape = (28,28,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255
y_train = keras.utils.to_categorical(y_train,classes)
y_test = keras.utils.to_categorical(y_test,classes)
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(classes,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.fit(x_train,y_train,batchsize,epochs=epochs)
result = model.evaluate(x_test,y_test)
print("test loss % and test accuracy %",(result[0],result[1]))
test = df_test.values
test.shape
test = test.reshape(test.shape[0],28,28,1)
input_shape = (28,28,1)
test = test.astype('float32')
test = test/255
pred = model.predict(test)
pred = model.predict_classes(test)
print(pred)
df_test.index
submission = pd.DataFrame({"ImageId":df_test.index+1,"Label":pred})
submission.to_csv("mnist_result.csv",index=False)
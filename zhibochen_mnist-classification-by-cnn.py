# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential,Model,load_model

from keras.layers import Dense,Dropout,Flatten,Input

from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D

from keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint

from keras.optimizers import RMSprop,Adam



from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

print("train shape:{}\ntest shape:{}".format(train.shape,test.shape))



train_df=train.drop(['label'],axis=1)

label=pd.get_dummies(train['label'])



#reshape image in 3 dimensions

train_df=train_df.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)



x_train,x_valid,y_train,y_valid=train_test_split(train_df,label,test_size=0.2,random_state=2017)

print('x_train:'+str(x_train.shape))

print('x_valid:'+str(x_valid.shape))
def VGG16():

    input=Input(shape=(28,28,1))

    

    x=Conv2D(64,(3,3))(input)

    x=Conv2D(64,(3,3))(x)

    x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    

    x=Conv2D(128,(3,3),activation='relu',padding='same')(x)

    x=Conv2D(128,(3,3),activation='relu',padding='same')(x)

    x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    

    x=Conv2D(256,(3,3),activation='relu',padding='same')(x)

    x=Conv2D(256,(3,3),activation='relu',padding='same')(x)

    x=Conv2D(256,(3,3),activation='relu',padding='same')(x)

    x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    

    x=Conv2D(512,(3,3),activation='relu',padding='same')(x)

    x=Conv2D(512,(3,3),activation='relu',padding='same')(x)

    x=Conv2D(512,(3,3),activation='relu',padding='same')(x)

    x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    

    x=Conv2D(512,(3,3),activation='relu',padding='same')(x)

    x=Conv2D(512,(3,3),activation='relu',padding='same')(x)

    x=Conv2D(512,(3,3),activation='relu',padding='same')(x)

    

    x=Flatten()(x)

    x=Dense(4096,activation='relu')(x)

    x=Dense(4096,activation='relu')(x)

    x=Dense(2018,activation='relu')(x)

    x=Dense(1000,activation='relu')(x)

    output=Dense(10,activation='softmax')(x)

    

    cnn_model=Model(input,output)

    

    return cnn_model

model=VGG16()



model.compile(optimizer=Adam(lr=1e-4),

             loss='categorical_crossentropy',

             metrics=['accuracy'])



earlystop=EarlyStopping(monitor='val_loss',patience=5)



history=model.fit(x_train,y_train,epochs=20,batch_size=64,verbose=2,

                  validation_data=(x_valid,y_valid),

                  callbacks=[earlystop,TensorBoard(log_dir='../log')])
import matplotlib.pyplot as plt



#trian & validation accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train','test'],loc='upper left')

plt.show()



#train & validation loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','test'],loc='upper left')

plt.show()
results=model.predict(test)

print(results)
#results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)



print(os.listdir("../input"))
import os
import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from keras.utils import np_utils
import pandas as pd
import numpy as np
train_data=pd.read_csv('../input/train.csv')
train_data.head()
x_train=train_data.values[:,1:]
y_train=train_data.values[:,0]
x_train=x_train.reshape((x_train.shape[0],28,28,1))
y_train=np_utils.to_categorical(y_train)

test_data=pd.read_csv('../input/test.csv')
x_test=test_data.values
x_test=x_test.reshape((x_test.shape[0],28,28,1))
input_shape=(28,28,1)
num_classes=10

model = Sequential()

model.add(BatchNormalization(axis=-1,input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),activation='relu',name="L1_1"))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="L1_2"))

model.add(Conv2D(22, kernel_size=(3, 3), strides=(1, 1),activation='relu',name="L2_1"))

model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="L2_2"))

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),activation='relu',name="L3_1"))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu',name="L3_2"))

model.add(Flatten())

model.add(BatchNormalization(axis=-1,name="L9"))

model.add(Dense(64, activation='relu',name="L10"))

model.add(Dense(40, activation='relu',name="L11_1"))

model.add(Dense(40, activation='relu',name="L11_2"))

model.add(Dense(32, activation='relu',name="L12"))

model.add(Dense(num_classes, activation='softmax',name="L13"))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(lr=0.003), metrics=['accuracy'])

model.summary()
model.fit(x_train,y_train,epochs=32,batch_size=512,verbose=2)
with open('predictions.csv','w') as outputFile:
    predictions=model.predict(x_test)
    outputFile.write("ImageId,Label\n")
    for index,probs in zip(range(len(predictions)),predictions):
        outputFile.write(str(index+1)+","+str(np.argmax(probs))+"\n")
    outputFile.close()
    print("saved output to predictions.csv")
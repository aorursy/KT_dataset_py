import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

"import matplotlib.image as mpimg"

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Lambda,Dense,Conv2D,Dropout,Flatten

from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop,SGD,Adagrad
from keras.utils import to_categorical
Data_Train=pd.read_csv("../input/train.csv")
Image_train,Image_val,Label_train,Label_val=train_test_split(Data_Train.iloc[:,1:],Data_Train.iloc[:,0:1],test_size=0.33,random_state=0)
def fromDFtoTensor(x):

    t=x.values.astype('float32')

    x=t.reshape(t.shape[0],28,28,1)

    return x

Image_train=fromDFtoTensor(Image_train)

Image_val=fromDFtoTensor(Image_val)

Label_train=Label_train.values.astype("int").ravel()

Label_val=Label_val.values.astype("int").ravel()
plt.imshow(Image_train[0].reshape(28,28),cmap="gray")
y_train=to_categorical(Label_train,10)

y_val=to_categorical(Label_val,10)
mean=Image_train.mean().astype(np.float32)

std=Image_train.std().astype(np.float32)
def standardizationImage(x):

    x-=mean

    x/=std

    return x
model=Sequential()

Input=(28,28,1)

model.add(Lambda(standardizationImage,input_shape=Input))

model.add(Conv2D(32,(3,3),activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))

model.add(Conv2D(64,(3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(10,activation="softmax"))





opt=RMSprop(lr=0.0001)

model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(Image_train,y_train,epochs=15,batch_size=32)
model.evaluate(Image_val,y_val,batch_size=32)
model.metrics_names
Data_test=pd.read_csv("../input/test.csv")
Image_test=fromDFtoTensor(Data_test)
results=model.predict_classes(Image_test,verbose=0)
{"ImageID":Data_test.index.values+1,"Label":results}
pd.DataFrame({"ImageID":Data_test.index.values+1,"Label":results}).to_csv("result.csv",index=False)
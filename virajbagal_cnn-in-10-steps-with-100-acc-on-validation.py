# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output

seed=5
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.label.value_counts().plot.bar()
train.label.value_counts()
df=train.groupby('label').apply(lambda x: x.sample(3795)).reset_index(drop=True)

df.label.value_counts()
df=df.sample(frac=1,random_state=seed)
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical



target=df['label']

X=df.drop('label',axis=1)

target=to_categorical(target)



train_X,val_X,train_y,val_y=train_test_split(X,target,test_size=0.01,random_state=seed)
train_X=train_X/255

val_X=val_X/255



train_X=train_X.values.reshape(-1,28,28,1)

val_X=val_X.values.reshape(-1,28,28,1)
import matplotlib.pyplot as plt

%matplotlib inline





plt.imshow(train_X[0][:,:,0])
from keras.preprocessing.image import ImageDataGenerator





train_datagen=ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,

                                rotation_range=10)



train_datagen.fit(train_X)
from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D,BatchNormalization,Dropout,Flatten,Dense

from keras.callbacks import EarlyStopping,ReduceLROnPlateau
model=Sequential()

model.add(Conv2D(64,3,input_shape=(28,28,1),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(2,padding='same'))

model.add(Conv2D(128,3,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(2,padding='same'))

model.add(Conv2D(256,3,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(2,padding='same'))

model.add(Conv2D(512,3,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(2,padding='same'))

model.add(Conv2D(1024,3,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(2,padding='same'))

model.add(Conv2D(1024,3,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(2,padding='same'))

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.8))

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.8))

model.add(Dense(10,activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
est=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

rlp=ReduceLROnPlateau(monitor='val_loss',patience=10,factor=0.1,min_delta=0.0001)

call_backs=[est,rlp]
#tune epochs to 30 for 100 percent accuracy on validation



batch_size=64

result=model.fit_generator(train_datagen.flow(train_X,train_y,batch_size=batch_size),epochs=1,

                  callbacks=call_backs,validation_data=(val_X,val_y),validation_steps=1,

                          steps_per_epoch=train_X.shape[0]//batch_size)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



plt.plot(result.history['loss'],label='Train loss')

plt.plot(result.history['val_loss'],label='Validation loss')

plt.legend(loc='best')
from sklearn.metrics import confusion_matrix



val_pred_probs=model.predict(val_X)

val_labels=np.argmax(val_pred_probs,axis=1)

true_labels=np.argmax(val_y,axis=1)

cm=confusion_matrix(val_labels,true_labels)



plt.figure(figsize=(10,6))

sns.heatmap(cm,annot=True)

test=test/255

test=test.values.reshape(-1,28,28,1)
test_probs=model.predict(test)

test_labels=np.argmax(test_probs,axis=1)
sub=pd.read_csv('../input/sample_submission.csv')

sub['label']=test_labels

sub.to_csv('Submission.csv',index=False)
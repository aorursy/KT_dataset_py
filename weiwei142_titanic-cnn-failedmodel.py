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
all_df = pd.read_csv('../input/train.csv')
from sklearn.preprocessing import StandardScaler 

def PreprocessData(all_df):

    df = all_df.drop(['Survived','Name','Ticket','Cabin'],axis=1)

    age_mean = df['Age'].mean() 

    df['Age'] = df['Age'].fillna(age_mean) #年紀當中有NaNull值（不詳），將平均值帶入

    fare_mean = df['Fare'].mean()

    df['Fare'] = df['Fare'].fillna(fare_mean) #票價同理

    df['Sex'] = df['Sex'].map({'female':0,'male':1}).astype(int)

    X_OneHot = pd.get_dummies(data=df,columns=['Embarked'])

    #標準化處理

    ss = StandardScaler()

    Features = ss.fit_transform(X_OneHot.values)

    

    Label = pd.get_dummies(all_df['Survived']).values

    #

    return Features, Label
np.random.seed(10)

msk = np.random.rand(len(all_df)) < 0.7

train_df = all_df[msk]

test_df = all_df[~msk]



x_train, y_train = PreprocessData(train_df)

x_test, y_test = PreprocessData(test_df) 



from keras.models import Sequential

from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten,Conv2D

from keras import optimizers



model = Sequential()

img_rows, img_cols = 5,2

x_train = np.reshape(x_train,(x_train.shape[0],img_rows, img_cols,1))

x_test = np.reshape(x_test,(x_test.shape[0],img_rows, img_cols,1))

input_size = (img_rows,img_cols,1)

from keras import backend as K

K.set_image_dim_ordering('tf')


model = Sequential()

model.add(Conv2D(filters=1000,input_shape=input_size, kernel_size = (3,1),strides=(1,1),padding='valid',activation='relu'))

model.add(Conv2D(filters=500,kernel_size=(3,1),strides=(1,1),padding='valid',activation='relu'))

model.add(MaxPooling2D(pool_size=(1,1)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=250,kernel_size = (1,1),strides=(1,1),padding='valid',activation='relu'))

model.add(MaxPooling2D(pool_size=(1,1)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(units=125,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=2,activation='softmax'))

model.summary()
model.compile(optimizer='Nadam',loss='binary_crossentropy',metrics=['accuracy'])

train_history = model.fit(x=x_train,y=y_train,validation_split=0.1,epochs=100,batch_size=30,verbose=2)
import matplotlib.pyplot as plt

def show_train_history(train_history, train, validation):

    plt.plot(train_history.history[train])

    plt.plot(train_history.history[validation])

    plt.title('Trian History')

    plt.ylabel(train)

    plt.xlabel("Epochs")

    plt.legend(['train','validation'],loc='lower right')

    plt.show()

show_train_history(train_history,'acc','val_acc')

show_train_history(train_history,'loss','val_loss')
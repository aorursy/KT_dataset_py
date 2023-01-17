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
import tensorflow as tf

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import SGD



train_data=pd.read_csv("/kaggle/input/disease-prediction-using-machine-learning/Training.csv")

test_data=pd.read_csv("/kaggle/input/disease-prediction-using-machine-learning/Testing.csv")



train_data.columns
y_train=np.array(train_data['prognosis'])

x_train=np.array(train_data.drop(columns=['prognosis','Unnamed: 133']))





y_test=np.array(test_data['prognosis'])

x_test=np.array(test_data.drop(columns=['prognosis']))
#np.unique([1, 1, 2, 2, 3, 3])

values=np.unique(y_train)

values
Y_train=np.zeros(shape=(len(y_train),len(values)))

k=0

for x in y_train:

     for i in range(len(values)):

            if x==values[i]:

                tmp=list(np.zeros(41))

                tmp[i]=1

                Y_train[k]=tmp

                k+=1

Y_train[0]
np.unique(Y_train)
model = Sequential()

model.add(Dense(4920, input_dim=132))

model.add(Activation('tanh'))

model.add(Dense(41))

model.add(Activation('sigmoid'))



sgd = SGD(lr=0.1)

model.compile(loss='binary_crossentropy',metrics=['accuracy'], optimizer=sgd)



model.fit(x_train, Y_train, batch_size=1, epochs=3)
#np.unique([1, 1, 2, 2, 3, 3])

test_values=np.unique(y_test)

test_values



Y_test=np.zeros(shape=(len(y_test),len(test_values)))

k=0

for x in y_test:

     for i in range(len(test_values)):

            if x==test_values[i]:

                tmp=list(np.zeros(41))

                tmp[i]=1

                Y_test[k]=tmp

                k+=1

pre=model.predict_proba(x_test)
pre[0].max()
acc=0;

for i in range(len(pre)):

    if pre[i].argmax() == Y_test[i].argmax() :

        if pre[i].max() >= 0.90:

            acc+=1



acc=acc/len(pre)

acc
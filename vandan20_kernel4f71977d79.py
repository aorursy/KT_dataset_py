# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import pandas as pd

sample=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

sample.head()
del sample
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

train.head()
y_train=train['label']

x_train=train.drop('label',axis=1)

del train
x_train=x_train/256
plt.imshow(x_train.iloc[64].values.reshape(28,28))
from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D



model=Sequential()



model.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=3,activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(256,activation='sigmoid'))

model.add(Dense(10,activation='softmax'))
from keras.utils import to_categorical

y_train=to_categorical(y_train)
x_train=x_train.to_numpy()

x_train=x_train.reshape(42000,28,28,1)
x_train.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.1)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=2)
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test=test/256

test=test.to_numpy()
test.shape
test=test.reshape(28000,28,28,1)

predictions=model.predict(test)
predictions
import numpy as np

labels=np.argmax(predictions,axis=1)

labels
sub=pd.DataFrame({'ImageId':range(1,28001),'Label':labels})

sub
test1=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test1
sub.to_csv('submission.csv')
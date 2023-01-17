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
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
dataframe=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv",sep=',')

dataframe=dataframe.reindex(np.random.permutation(dataframe.index))

dataset=dataframe.values

train_dataset=dataset[:,0:8].astype(float)

test_dataset=dataset[:,8]
model=Sequential()

model.add(Dense(9,input_dim=8,activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(768,activation='sigmoid'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_dataset,test_dataset,epochs=200,batch_size=20)
scores=model.evaluate(train_dataset,test_dataset)

print("\n%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))

print(dataframe.head())
predictions=model.predict(train_dataset)

predictions=predictions.astype(float)
print(predictions)
print(test_dataset)
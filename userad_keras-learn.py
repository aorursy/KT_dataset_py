# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from pylab import rcParams

rcParams['figure.figsize'] = 12,10



#Ok, now neural time

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential

from keras.layers import Dense,Dropout,Activation

from keras.utils.np_utils import to_categorical
data = pd.read_csv('../input/train.csv')

orig_data = data.copy()

ids = data.pop('id')

data_y = data.pop('species')

data_y_encoded = LabelEncoder().fit(data_y).transform(data_y)

data_x = StandardScaler().fit(data).transform(data)

y_cat = to_categorical(data_y_encoded)
model = Sequential()

model.add(Dense(1024, input_dim=data_x.shape[1]))

model.add(Activation('sigmoid'))

model.add(Dense(512))

model.add(Activation('sigmoid'))

model.add(Dense(y_cat.shape[1]))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
train_iterations = model.fit(data_x, y_cat, batch_size=128, nb_epoch=100, verbose=0)
plt.plot(train_iterations.history['loss'])

plt.xlabel('Number of Iterations')

plt.ylabel('Categorical Crossentropy')

plt.title('Train Error')

plt.yscale('log')
test = pd.read_csv('../input/test.csv')

test_ids = test.pop('id')

test_x = StandardScaler().fit(test).transform(test)

test_y = model.predict_proba(test_x)

submission = pd.DataFrame(test_y,index=test_ids,columns=orig_data.species.unique())

submission.to_csv('submission8.csv')
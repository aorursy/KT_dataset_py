# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.contrib.keras import models

from tensorflow.contrib.keras import layers

from tensorflow.contrib.keras import losses,optimizers,metrics



%matplotlib inline

sns.set_style('whitegrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
voice = pd.read_csv('../input/voice.csv')



print(voice.columns)

voice.head()
voice.describe()
voice = pd.get_dummies(voice)

voice.drop('label_male',axis=1,inplace=True)

voice.head()
# change label if you want

voice['label'] = voice['label_female']

voice.drop('label_female',axis=1,inplace=True)
plt.figure(figsize=[16,9])

mask = np.ones_like(voice.corr())

mask[np.tril_indices_from(mask)] = False

sns.heatmap(voice.corr(),mask=mask,vmin=-1,vmax=1,cmap='coolwarm',annot=True,linewidths=0.5)
voice.drop('centroid',axis=1,inplace=True)
plt.figure(figsize=[9,6])

sns.boxplot(x='label',y='meanfreq',data=voice)

plt.xticks([0,1],['male','female'])

plt.xlabel(xlabel=None)
plt.figure(figsize=[9,6])

sns.boxplot(x='label',y='meanfun',data=voice)

plt.xticks([0,1],['male','female'])

plt.xlabel(xlabel=None)
voice_data = voice.drop('label',axis=1)

voice_label = voice['label']
X_train, X_test, y_train, y_test = train_test_split(voice_data,voice_label,test_size=0.3)
scaler = MinMaxScaler()

# only train the scaler on the training data

scaled_x_train = scaler.fit_transform(X_train)

scaled_x_test = scaler.transform(X_test)



print('Scaled training data shape: ',scaled_x_train.shape)
dnn_keras_model = models.Sequential()
# can play around with the number of hidden layers but I found that one hidden layer was more than enough to give great metrics

dnn_keras_model.add(layers.Dense(units=30,input_dim=19,activation='relu'))

# dnn_keras_model.add(layers.Dense(units=30,activation='relu'))

dnn_keras_model.add(layers.Dense(units=20,activation='relu'))

dnn_keras_model.add(layers.Dense(units=10,activation='relu'))

dnn_keras_model.add(layers.Dense(units=2,activation='softmax'))
# compile model by selecting optimizer and loss function

dnn_keras_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# train/fit the model

dnn_keras_model.fit(scaled_x_train,y_train,epochs=50)
predictions = dnn_keras_model.predict_classes(scaled_x_test)
print('Metric for ')

print('Classification report:')

print(classification_report(predictions,y_test))

print('\n')

print('Confusion matrix:')

print(confusion_matrix(predictions,y_test))

print('\n')

print('Accuracy score is {:6.3f}.'.format(accuracy_score(predictions,y_test)))
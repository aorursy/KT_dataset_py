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
df= pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")
df.info()
df= df.drop(columns=["Evaporation","Sunshine","Cloud9am","Cloud3pm"],axis=1)

df.shape
df = df.drop(columns=["Date","Location"],axis=1)

df.shape
df.isnull().sum()
df= df.dropna(how="any")

df.shape
from sklearn.preprocessing import LabelEncoder

labelencoder_df = LabelEncoder()

df['WindGustDir']= labelencoder_df.fit_transform(df['WindGustDir'])

df['WindDir9am']= labelencoder_df.fit_transform(df['WindDir9am'])

df['WindDir3pm']= labelencoder_df.fit_transform(df['WindDir3pm'])

df['RainToday']= labelencoder_df.fit_transform(df['RainToday'])

df['RainTomorrow']= labelencoder_df.fit_transform(df['RainTomorrow'])
df.head()

df.tail()
import matplotlib.pyplot as plt

import seaborn as sns

corr= df.corr()

plt.figure(figsize=(12,10))

sns.heatmap(corr,xticklabels= corr.columns.values,yticklabels= corr.columns.values,annot= True,fmt='.2f',linewidth=0.30)
x=df.iloc[:,0:17].values

y=df.iloc[:,-1].values
x.shape

y.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler

sc_x= StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.fit_transform(x_test)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators= 200, random_state=0)

classifier.fit(x_train,y_train)
y_pred= classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score

cm= confusion_matrix(y_test,y_pred)

accuracy = accuracy_score(y_test,y_pred)

print("RandomForestClassification:")

print("Accuracy = ",accuracy)

print(cm)
from collections import Counter

Counter(y_train)
Counter(y_test)
Counter(y_pred)
df['meantemp'] = df.loc[: , "MinTemp":"MaxTemp"].mean(axis =1)

df = df.drop(['MinTemp', 'MaxTemp','RainToday','RainTomorrow'], axis=1)
import matplotlib.pyplot as plt

import seaborn as sns

corr= df.corr()

plt.figure(figsize=(12,10))

sns.heatmap(corr,xticklabels= corr.columns.values,yticklabels= corr.columns.values,annot= True,fmt='.2f',linewidth=0.30)
X = df.drop(['meantemp'], axis=1)

y = df['meantemp']

X= StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print('X_train : ',X_train.shape)

print('y_train : ',y_train.shape)

print('X_test : ',X_test.shape)

print('y_test : ',y_test.shape)
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

import sklearn

from sklearn import preprocessing
#applying the neurons in the layers with weights and biases to work on the algorithm....linear regression is applied... y =mx_c.

model = Sequential()

model.add(Dense(13, input_shape=(14,), activation='relu'))

model.add(Dense(13, activation='relu'))

model.add(Dense(13, activation='relu'))

model.add(Dense(13, activation='relu'))

model.add(Dense(13, activation='relu'))

model.add(Dense(1,))

model.compile(Adam(lr=0.003), 'mean_squared_error',metrics=['mse', 'mae', 'mape', 'cosine'])

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# Runs model for 2000 iterations and assigns this to 'history'

model.summary()

history = model.fit(X_train, y_train, epochs = 100,validation_split = 0.2, verbose = 1)
from sklearn import model_selection

from sklearn.metrics import r2_score

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)

print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))

print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))
history_dict=history.history

loss_values = history_dict['loss']

val_loss_values=history_dict['val_loss']

plt.title('model Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.plot(loss_values,'bo',label='training loss')



plt.plot(val_loss_values,'r',label='validation loss ')

plt.legend(['train', 'validation'], loc='upper right')
from matplotlib import pyplot



pyplot.plot(history.history['mse'])

pyplot.plot(history.history['mae'])

pyplot.plot(history.history['mape'])

pyplot.plot(history.history['cosine'])



plt.legend(['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'], loc='upper right')

pyplot.show()
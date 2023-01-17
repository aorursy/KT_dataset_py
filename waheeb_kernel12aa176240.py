import numpy as np

import time

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import rcParams

%matplotlib inline

rcParams['figure.figsize'] = 10,8

sns.set(style='whitegrid', palette='muted',

        rc={'figure.figsize': (15,10)})

import os

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



from numpy.random import seed

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
trainX = pd.read_csv("../input/trainHome.csv")

trainX = trainX.drop("Unnamed: 0", axis=1)

testX = pd.read_csv("../input/testHome.csv")

testX = testX.drop("Unnamed: 0", axis=1)

#reading data from csv file and deleting the empty columns 
trainX.head()

#data sample 
scaler = StandardScaler()

continuous = ['rate(1-10)', 'area m2'] #the data we want to scale 

for var in continuous: # train data

    trainX[var] = trainX[var].astype('float64') 

    trainX[var] = scaler.fit_transform(trainX[var].values.reshape(-1, 1))



for var in continuous: # test data

    testX[var] = testX[var].astype('float64')

    testX[var] = scaler.fit_transform(testX[var].values.reshape(-1, 1))    
trainX.head() # data sample after scaling 

trainY = trainX[pd.notnull(trainX['price (jd)'])]['price (jd)'] #spliting data from the goal

trainX = trainX[pd.notnull(trainX['price (jd)'])].drop(['price (jd)'], axis=1)



testY = testX[pd.notnull(testX['price (jd)'])]['price (jd)'] #spliting data from the goal

testX = testX[pd.notnull(testX['price (jd)'])].drop(['price (jd)'], axis=1)
def create_model():

    

    

    model = Sequential()

    

    model.add(Dense(256, activation='relu',init='normal',input_dim=trainX.shape[1]))

    

    model.add(Dense(256, activation='relu'))

    model.add(Dense(256, activation='relu'))

    model.add(Dense(256, activation='relu'))

    model.add(Dense(256, activation='relu'))

    model.add(Dense(256, activation='relu'))

    model.add(Dense(256, activation='relu'))

    model.add(Dense(256, activation='relu'))

    model.add(Dense(256, activation='relu'))



   

   

    model.add(Dense(1,init='normal'))  

    

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    

    return model
model = create_model() 

start = time.time() 

training = model.fit(trainX, trainY, epochs=500, batch_size=2, validation_split=0.25, verbose=0, validation_data=(testX, testY))

end = time.time()

print(end - start)
plt.plot(training.history['mean_absolute_error'])

plt.plot(training.history['val_mean_absolute_error'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
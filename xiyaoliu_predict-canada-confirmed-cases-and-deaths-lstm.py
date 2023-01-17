# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot

import seaborn as sns; sns.set()

# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import missingno as msno

# Configure visualisations

%matplotlib inline

mpl.style.use( 'ggplot' )

plt.style.use('fivethirtyeight')

sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import csv as csv

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Download Data

deaths_global = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv')

confirmed_global = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')
deaths_global.head()
confirmed_global.head()
deaths_global.tail()
confirmed_global.tail()
country=['Canada', 'US']

y=deaths_global.loc[deaths_global['Country/Region']=='Italy'].iloc[0,4:]

s = pd.DataFrame({'Italy':y})

for i in country:    

    s[i] = deaths_global.loc[deaths_global['Country/Region']==i].iloc[0,4:]

pyplot.plot(range(y.shape[0]), s)
country=['Canada', 'US']

y=confirmed_global.loc[confirmed_global['Country/Region']=='Italy'].iloc[0,4:]

s = pd.DataFrame({'Italy':y})

for i in country:    

    s[i] = confirmed_global.loc[confirmed_global['Country/Region']==i].iloc[0,4:]

pyplot.plot(range(y.shape[0]), s)
cols = confirmed_global.keys()

cols
cols_d=deaths_global.keys()

cols_d
confirmed_g = confirmed_global.loc[:, cols[4]:cols[-1]]

confirmed_g
deaths_g = deaths_global.loc[:, cols[4]:cols[-1]]

deaths_g
confirmed_canada = confirmed_global.loc[confirmed_global['Country/Region']=='Canada'].iloc[:,4:].values
deaths_canada = deaths_global.loc[deaths_global['Country/Region']=='Canada'].iloc[:,4:].values
confirmed_canada
confirmed_canada.shape
deaths_canada
deaths_canada.shape
dates = confirmed_g.keys()

df1 = pd.DataFrame(confirmed_canada,columns=dates)

df1
dates_d = deaths_g.keys()

df2 = pd.DataFrame(deaths_canada,columns=dates_d)

df2
#Caculate the sum of confirmed cases for each day

ca_sum= df1.sum(axis=0)

ca_sum=pd.DataFrame(ca_sum) 

ca_sum
#Caculate the sum of deaths for each day

ca_sum_d= df2.sum(axis=0)

ca_sum_d=pd.DataFrame(ca_sum_d) 

ca_sum_d
ca_training_processed = ca_sum.iloc[:, 0:1].values

#ca_training_processed
ca_d_training_processed = ca_sum_d.iloc[:, 0:1].values
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))



ca_training_scaled = scaler.fit_transform(ca_training_processed)



ca_d_training_scaled = scaler.fit_transform(ca_d_training_processed)
#Convert Training Data to Right Shape for the cnfirmed cases' dataset

features_set = []

labels = []

for i in range(15, 87):

    features_set.append(ca_training_scaled[i-15:i, 0])

    labels.append(ca_training_scaled[i, 0])
#Convert Training Data to Right Shape for the deaths' dataset

features_set_d = []

labels_d = []

for i in range(15, 87):

    features_set_d.append(ca_d_training_scaled[i-15:i, 0])

    labels_d.append(ca_d_training_scaled[i, 0])
#X_train and y_train for confirmed cases

features_set, labels = np.array(features_set), np.array(labels)
#X_train and y_train for deaths

features_set_d, labels_d = np.array(features_set_d), np.array(labels_d)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

#features_set = np.reshape(features_set, (72, 15, 1))

features_set_d = np.reshape(features_set_d, (features_set_d.shape[0], features_set_d.shape[1], 1))
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

#Training The LSTM for confirmed cases

#Creating LSTM and Dropout Layers

model = Sequential()

model.add(LSTM(units=256, activation='relu', return_sequences=True, input_shape=(features_set.shape[1], 1)))

model.add(Dropout(0.2))

model.add(LSTM(units=128, return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(units=128, return_sequences=False))

model.add(Dropout(0.2))

#model.add(LSTM(units=64))

#model.add(Dropout(0.2))



model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(features_set, labels, epochs = 100, batch_size = 32)
#Training The LSTM for deaths



model_d = Sequential()

model_d.add(LSTM(units=256, activation='relu', return_sequences=True, input_shape=(features_set_d.shape[1], 1)))

model_d.add(Dropout(0.2))

model_d.add(LSTM(units=128, return_sequences=True))

model_d.add(Dropout(0.2))

model_d.add(LSTM(units=128, return_sequences=False))

model_d.add(Dropout(0.2))



model_d.add(Dense(units = 1))

model_d.compile(optimizer = 'adam', loss = 'mean_squared_error')



model_d.fit(features_set_d, labels_d, epochs = 60, batch_size = 32)
# Define the function to predict the number of comfirmed cases in the next days

def predict_comfirmed_CA_model(number_days):

    features=ca_training_scaled.copy()

    pred_data=[]

    for i in range(number_days-1):

        features=features[-15:]

        #print(features.shape[1])

        features = features.reshape(1,15, 1)

        prediction_record=model.predict(features, verbose=0)

        #print(prediction_record)

        pred_data=np.append(pred_data,prediction_record)

        features= np.append(features, prediction_record)

    return  pred_data 

# Define the function to predict the number of deaths in the next days

def predict_deaths_CA_model(number_days):

    features_d=ca_d_training_scaled.copy()

    pred_d_data=[]

    for i in range(number_days-1):

        features_d=features_d[-15:]

        features_d = features_d.reshape(1,15, 1)

        prediction_d_record=model_d.predict(features_d, verbose=0)

        #print(prediction_record)

        pred_d_data=np.append(pred_d_data,prediction_d_record)

        features_d= np.append(features_d, prediction_d_record)

    return  pred_d_data 
# Predict the confirmed cases in the next 15 days

pred15=predict_comfirmed_CA_model(15)

pred15
# Plot to the prediction of confirmed cases

plt.plot(np.append(ca_training_scaled,pred15))

plt.axis('tight')

plt.show()
index=list(range(87,87+14))

plt.plot(ca_training_scaled)

plt.plot(index,pred15)

plt.show()
# Predict the the number of deaths in the next 15 days

pred15_d=predict_deaths_CA_model(15)

pred15_d
# Plot to the prediction of deaths

plt.plot(np.append(ca_d_training_scaled,pred15_d))

plt.axis('tight')

plt.show()
index_d=list(range(87,87+14))

plt.plot(ca_d_training_scaled)

plt.plot(index_d,pred15_d)

plt.show()
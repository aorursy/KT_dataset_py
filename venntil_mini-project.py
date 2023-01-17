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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler
#Load data

data = pd.read_csv('../input/train.csv')

data.head()
#Load test

test = pd.read_csv('../input/test.csv')

test.head()
print('data shape: ', data.shape)

print('test shape: ', test.shape)
#Check for null rows

print('data null rows\n', data.isna().sum())

print()

print('data null rows\n', test.isna().sum())
#Visualize raw data

data.boxplot(column='SalePrice', by='OverallQual', figsize = (10, 10))
data.boxplot(column='SalePrice', by='BldgType', figsize = (10, 10))
plt.figure(figsize=(10, 10))

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.scatter(data['GrLivArea'], data['SalePrice'])

plt.show()
plt.figure(figsize=(10, 10))

plt.xlabel('GarageArea')

plt.ylabel('SalePrice')

plt.scatter(data['GarageArea'], data['SalePrice'])

plt.show()
print('Saleprice \n min:', data['SalePrice'].min(), '\n max:', data['SalePrice'].max(), '\n mean:', data['SalePrice'].mean(), '\n variance:', data['SalePrice'].var())
#1-hot BldgType for data

data['1Fam'] = (data['BldgType'] == '1Fam') * 1.0

data['2FmCon'] = (data['BldgType'] == '2FmCon') * 1.0

data['Duplx'] = (data['BldgType'] == 'Duplx') * 1.0

data['TwnhsE'] = (data['BldgType'] == 'TwnhsE') * 1.0

data['TwnhsI'] = (data['BldgType'] == 'TwnhsI') * 1.0

data.drop(['Id', 'BldgType'], axis = 1, inplace = True)

data.head()
#1-hot BldgType for test

test['1Fam'] = (test['BldgType'] == '1Fam') * 1.0

test['2FmCon'] = (test['BldgType'] == '2FmCon') * 1.0

test['Duplx'] = (test['BldgType'] == 'Duplx') * 1.0

test['TwnhsE'] = (test['BldgType'] == 'TwnhsE') * 1.0

test['TwnhsI'] = (test['BldgType'] == 'TwnhsI') * 1.0

test.drop(['Id', 'BldgType'], axis = 1, inplace = True)

test.head()
# Split data into training(features) data & target(saleprice) data

train = data.drop('SalePrice', axis = 1)

target = data['SalePrice']



#normalize training data

scale = MinMaxScaler()

train = scale.fit_transform(train)

#scale = MinMaxScaler()

#target = target.values.reshape(-1,1)

#target = scale.fit_transform(target)



x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.33, random_state=13)
print('training data', x_train.shape, 'validation data', x_test.shape)

print('training target', y_train.shape, 'validation target', y_test.shape)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import metrics





model = keras.Sequential()

model.add(layers.Dense(32, input_dim = 8, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

#model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(1))



#optimizier = keras.optimizers.SGD(lr=0.01, momentum= 0.5, nesterov=False)

optimizier = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer = optimizier, loss = 'mean_squared_error', metrics =[metrics.mae])

model.summary()

#loss metric is mse, added mae to visualize
#train model

history = model.fit(x_train, y_train, epochs = 500, validation_data = (x_test, y_test), verbose = 0)
#Check results

hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch



hist.dtypes
plt.figure()

plt.xlabel('Epochs')

plt.ylabel('MAE')

plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train error')

plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val error')

plt.legend()
#run model on validation

eva = model.evaluate(x_test, y_test)

print(model.metrics_names)

print(eva[1])
#run model on submission data

test.head()
scale = MinMaxScaler()

test_scaled = scale.fit_transform(test)
pred = model.predict(test_scaled)

test['SalePrice'] = pred

test.head()
sub = pd.read_csv('../input/submission.csv')

sub['SalePrice'] = pred

sub.head()
sub.to_csv('Testing Results.csv')
#Linear regression with sklearn

reg = LinearRegression(normalize = False)

reg.fit(x_train, y_train)



x = reg.predict(x_test)



plt.figure(figsize=(10, 10))

plt.xlabel('Actual')

plt.ylabel('Prediction')

plt.scatter(y_test, x, label='Train acc')

plt.plot([0,y_test.max()], [0,y_test.max()])



plt.show()
mae = abs(x - y_test)

mae.mean()
#import pandas as pd

#submission = pd.read_csv("../input/submission.csv")

#test = pd.read_csv("../input/test.csv")

#train = pd.read_csv("../input/train.csv")
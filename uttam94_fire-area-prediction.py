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
!ls ../input/forest-forest-dataset
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/forest-forest-dataset/forestfires.csv')
# checking columns of dataset 

df.columns
df.head()
# checking null values 

df.isnull().sum()
plt.figure(figsize=(10,10))

corr= df.corr()

sns.heatmap(corr,annot=True)
df.dtypes

#month and days are object type hence applying labelencoder and onehotencoder  
# convering categorical data 

df = pd.get_dummies(df, prefix=['month','day'],drop_first=True)
df.shape
y = df.iloc[:,[27]].values

x = df.iloc[:,:-1].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y ,test_size = 0.25,random_state= 42)
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(x, y)
print(y_test.shape, y_train.shape)
x.shape
y_pred = regressor.predict(x_test)
from sklearn import metrics

print('mean_absolute_error : {} '.format(metrics.mean_absolute_error(y_test, y_pred)))

print('mean_squared_error : {} '.format(metrics.mean_squared_error(y_test, y_pred)))

print('mean_squared_error : {} '.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
# checking current parameter used in random forest regressor 

regressor.get_params
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]
# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(x,y)
rf_random.best_params_
hyper_regressor = RandomForestRegressor(n_estimators=800,min_samples_split=2,min_samples_leaf=2,max_features='sqrt',max_depth=20,bootstrap=False)

hyper_regressor.fit(x, y)
y_pred = hyper_regressor.predict(x_test)
# checking mean absolute error , mean square error , RMSE

from sklearn import metrics

print('mean_absolute_error : {} '.format(metrics.mean_absolute_error(y_test, y_pred)))

print('mean_squared_error : {} '.format(metrics.mean_squared_error(y_test, y_pred)))

print('root_mean_squared_error : {} '.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
# importing required librabries 

import keras 

from keras.models import Sequential

from keras.layers import Dense ,Flatten

from keras.layers import Dropout

from keras.layers import LeakyReLU,PReLU,ELU
#intilaizing the ANN

ann_regressor = Sequential()

# adding input layer or first hiden layer to regressor 

ann_regressor.add(Dense(output_dim=50,init = 'he_uniform',activation='relu',input_dim =27))

# Adding the second hidden layer

ann_regressor.add(Dense(output_dim = 25, init = 'he_uniform',activation='relu'))

# Adding the third hidden layer

ann_regressor.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu'))

# adding output layer 

# The Output Layer :

ann_regressor.add(Dense(1, init= 'he_uniform',activation='linear'))

# Compile the network :

ann_regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

ann_regressor.summary()
model_hist= ann_regressor.fit(x_train,y_train,validation_split=0.20, batch_size = 10, nb_epoch = 50)
ann_pred = ann_regressor.predict(x_test)
print('mean_absolute_error : {} '.format(metrics.mean_absolute_error(y_test, ann_pred)))

print('mean_squared_error : {} '.format(metrics.mean_squared_error(y_test, ann_pred)))

print('mean_squared_error : {} '.format(np.sqrt(metrics.mean_squared_error(y_test, ann_pred))))
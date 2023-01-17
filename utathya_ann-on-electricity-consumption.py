# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def datetounix(df):

    # Initialising unixtime list

    unixtime = []

    

    # Running a loop to convert Date to seconds

    for date in df['datetime']:

        unixtime.append(time.mktime(date.timetuple()))

    

    # Replacing Date with unixtime list

    df['datetime'] = unixtime

    return(df)
# import libs

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

import time

from sklearn.ensemble import ExtraTreesClassifier

import operator

from sklearn.preprocessing import StandardScaler

import keras

from keras.models import Sequential

from keras.layers import Dense





# read train dataframe

# file_path = os.path.join(os.path.abspath(''), 'train.csv')

df_train = pd.read_csv("../input/train.csv", encoding='ISO-8859-1', engine='c')



# read test dataframe

# file_path = os.path.join(os.path.abspath(''), 'test.csv')

df_test = pd.read_csv("../input/test.csv", encoding='ISO-8859-1', engine='c')

df_train.info()
# Converting to datetime

df_train['datetime'] = pd.to_datetime(df_train['datetime'])

df_test['datetime'] = pd.to_datetime(df_test['datetime'])

df_test.info()
# Creating features from DateTime for train data



df_test['Weekday'] = [datetime.weekday(date) for date in df_test.datetime]

df_test['Year'] = [date.year for date in df_test.datetime]

df_test['Month'] = [date.month for date in df_test.datetime]

df_test['Day'] = [date.day for date in df_test.datetime]

df_test['Time'] = [((date.hour*60+(date.minute))*60)+date.second for date in df_test.datetime]

df_test['Week'] = [date.week for date in df_test.datetime]

df_test['Quarter'] = [date.quarter for date in df_test.datetime]



# Creating features from DateTime for test data



df_train['Weekday'] = [datetime.weekday(date) for date in df_train.datetime]

df_train['Year'] = [date.year for date in df_train.datetime]

df_train['Month'] = [date.month for date in df_train.datetime]

df_train['Day'] = [date.day for date in df_train.datetime]

df_train['Time'] = [((date.hour*60+(date.minute))*60)+date.second for date in df_train.datetime]

df_train['Week'] = [date.week for date in df_train.datetime]

df_train['Quarter'] = [date.quarter for date in df_train.datetime]
# Create Dummy Variables for Train set

df_train.loc[df_train.var2 == 'A', 'var2A'] = 1

df_train.loc[df_train.var2 == 'B', 'var2B'] = 1



df_train['var2A'].fillna(0, inplace=True)

df_train['var2B'].fillna(0, inplace=True)



df_train.drop(['var2'], axis=1, inplace=True)



# Create Dummy Variables for Test set

df_test.loc[df_test.var2 == 'A', 'var2A'] = 1

df_test.loc[df_test.var2 == 'B', 'var2B'] = 1



df_test['var2A'].fillna(0, inplace=True)

df_test['var2B'].fillna(0, inplace=True)



df_test.drop(['var2'], axis=1, inplace=True)



# Creating X_test

X_test = datetounix(df_test).drop(['ID'], axis=1).values



# Remove target column from the df

df_train_features = df_train.drop(['electricity_consumption', 'ID'], axis=1)



# Convet timestamp to seconds

df_train_features = datetounix(df_train_features)



# store features in X array

X = df_train_features.values

y = df_train['electricity_consumption'].values
# create an instance for tree feature selection

tree_clf = ExtraTreesClassifier()



# fit the model

tree_clf.fit(X, y)



# Preparing variables

importances = tree_clf.feature_importances_

feature_names = df_train_features.columns.tolist()



feature_imp_dict = dict(zip(feature_names, importances))

sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)



indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    print("feature %d : %s (%f)" % (indices[f], sorted_features[f][0], sorted_features[f][1]))



# Plot the feature importances of the forest

plt.figure(0)

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", align="center")

plt.xticks(range(X.shape[1]), indices)

plt.xlim([-1, X.shape[1]])

plt.show()
############ Data Scaling ###################

sc = StandardScaler()

X = sc.fit_transform(X)

X_test = sc.transform(X_test)



y_norm = (y - min(y))/(max(y) - min(y))

y_norm
# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 14))



# Adding the second hidden layer

classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])



# Fitting the ANN to the training set

classifier.fit(X, y_norm, batch_size = 10, epochs = 100)



# Part 3 - Making the predictions and evaluating the model



# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred * (max(y) - min(y))) + min(y)



predictions = [int(elem) for elem in y_pred]



df_solution = pd.DataFrame()

df_solution['ID'] = df_test.ID



# Prepare Solution dataframe

df_solution['electricity_consumption'] = predictions

df_solution['electricity_consumption'].unique()



df_solution
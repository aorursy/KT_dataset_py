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

    

    # Running a loop for converting Date to seconds

    for date in df['DateTime']:

        unixtime.append(time.mktime(date.timetuple()))

    

    # Replacing Date with unixtime list

    df['DateTime'] = unixtime

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

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
# read train dataframe

# file_path = os.path.join(os.path.abspath(''), '../input/train_aWnotuB.csv')

df_train = pd.read_csv('../input/train_aWnotuB.csv', encoding='ISO-8859-1', engine='c')



# read test dataframe

# file_path = os.path.join(os.path.abspath(''), '../input/test_BdBKkAj.csv')

df_test = pd.read_csv('../input/test_BdBKkAj.csv', encoding='ISO-8859-1', engine='c')

df_test.info()
# Converting to datetime

df_train['DateTime'] = pd.to_datetime(df_train['DateTime'])

df_test['DateTime'] = pd.to_datetime(df_test['DateTime'])

df_test.info()
# Creating features from DateTime for train data



df_test['Weekday'] = [datetime.weekday(date) for date in df_test.DateTime]

df_test['Year'] = [date.year for date in df_test.DateTime]

df_test['Month'] = [date.month for date in df_test.DateTime]

df_test['Day'] = [date.day for date in df_test.DateTime]

df_test['Time'] = [((date.hour*60+(date.minute))*60)+date.second for date in df_test.DateTime]

df_test['Week'] = [date.week for date in df_test.DateTime]

df_test['Quarter'] = [date.quarter for date in df_test.DateTime]



# Creating features from DateTime for test data



df_train['Weekday'] = [datetime.weekday(date) for date in df_train.DateTime]

df_train['Year'] = [date.year for date in df_train.DateTime]

df_train['Month'] = [date.month for date in df_train.DateTime]

df_train['Day'] = [date.day for date in df_train.DateTime]

df_train['Time'] = [((date.hour*60+(date.minute))*60)+date.second for date in df_train.DateTime]

df_train['Week'] = [date.week for date in df_train.DateTime]

df_train['Quarter'] = [date.quarter for date in df_train.DateTime]
# create an instance for tree feature selection

tree_clf = ExtraTreesClassifier()



# first create arrays holding input and output data

# get the features into an array X

# remove target column from the df

df_train_features = df_train.drop(['Vehicles'], axis=1)



# Convet timestamp to seconds

df_train_features = datetounix(df_train_features)



# store features in X array

X = df_train_features.values



# Store target feature in y array

y = df_train['Vehicles'].values



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
# Visualising the histogram for positive reviews only from train and dataset

data = df_train.Vehicles

binwidth = 1

plt.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), log=False)

plt.title("Gaussian Histogram")

plt.xlabel("Traffic")

plt.ylabel("Number of times")

plt.show()
######################################## X_test creation for Prediction #################################



# Convert timestamp to seconds

df_test_features = datetounix(df_test.drop(['Year', 'Quarter', 'Month', 'ID'], axis=1))



# Create X_test from the test set



X_test = df_test_features.values



######################################## Dropping Features from train set #######################



df_train_features = df_train.drop(['Vehicles','Year', 'Quarter', 'Month', 'ID'], axis=1)



# Convert timestamp to seconds

df_train_features = datetounix(df_train_features)



# store features in X array

X = df_train_features.values



# store target in y array

y = df_train['Vehicles'].values
# Data prep

df_solution = pd.DataFrame()

df_solution['ID'] = df_test.ID



# Starting time for time calculations

start_time = time.time()



# Create decision tree object

clf = DecisionTreeClassifier(criterion='gini', random_state = 13)



# fit the model

clf.fit(X, y)



# predict the outcome for testing data

predictions = clf.predict(X_test)



print("The time taken to execute is %s seconds" % (time.time() - start_time))



# Prepare Solution dataframe

df_solution['Vehicles'] = predictions

df_solution
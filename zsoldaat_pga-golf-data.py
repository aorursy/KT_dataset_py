# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')



import random

import math

from scipy import stats

from sklearn import linear_model, model_selection, preprocessing, datasets



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
full_data = pd.read_csv('/kaggle/input/pga-tour-20102018-data/PGA_Data_Historical.csv')
full_data.head()
#Checking which statistics have null values in the original data. I don't plan on using any of these statistics to predict driving distance, so no need to drop null values as they won't be used anyway. 

full_data[full_data['Value'].isnull()]['Statistic'].unique()
full_data.info()
full_data['Statistic'].unique()[1:20]#Only showing the first 20 so that this does not take up too much space in the final write-up. I combed through all of these originally .
data = full_data[full_data['Statistic'].isin(['Driving Distance', 'Driving Accuracy Percentage', 'Club Head Speed', 'Ball Speed', 'Smash Factor', 'Launch Angle', 'Spin Rate', 'Distance to Apex', 'Apex Height', 'Hang Time', 'Carry Distance', 'Carry Efficiency', 'Total Distance Efficiency', 'Total Driving Efficiency'])]

data = data[data['Variable'].isin(['Driving Distance - (AVG.)', 'Driving Accuracy Percentage - (%)', 'Club Head Speed - (AVG.)', 'Ball Speed - (AVG.)', 'Smash Factor - (AVG.)', 'Launch Angle - (AVG.)', 'Spin Rate - (AVG.)', 'Distance to Apex - (AVG.)', 'Apex Height - (AVG.)', 'Hang Time - (AVG.)', 'Carry Distance - (AVG.)', 'Carry Efficiency - (AVG.)', 'Total Distance Efficiency - (AVG.)'])]
#Replacing the "'" with "f" so it will be properly interpreted by my conversion function

data['Value'] = data['Value'].map(lambda x:x.replace('\'', 'f'))



#Removing Commas

data['Value'] = data['Value'].map(lambda x:x.replace(',', ''))



#Function to convert feet and inches to metric

def to_meters(string):

    if 'f' not in string:

        return string

    

    else:    

        feet = int(string.split('f')[0])

        inches = int(string.split('f')[1].split('"')[0].strip())



        return 0.3048*feet + 0.0254*inches



#Converting imperial units to metric

data['Value'] = data['Value'].map(lambda x:to_meters(x))



#Finally, converting all values from string to numerical

data['Value'] = pd.to_numeric(data['Value'])



data.info()
pivot = pd.pivot_table(data = data, index = 'Player Name', columns = 'Statistic', values = 'Value')

pivot.head()
fig, ax = plt.subplots(4,3, figsize=(12,12))

y_index=0

x_index=0

for variable in pivot.drop(columns = 'Driving Distance').columns:

    sns.scatterplot(data = pivot, x=variable, y='Driving Distance', ax=ax[x_index, y_index]).set(ylabel = '')

    x_index += 1

    if x_index == 4:

        y_index += 1

        x_index = 0



plt.subplots_adjust(hspace = 0.3)



fig, ax = plt.subplots(4,3, figsize=(12,12))

y_index=0

x_index=0

for variable in pivot.drop(columns = 'Driving Distance').columns:

    sns.distplot(a=pivot[variable], ax=ax[x_index, y_index]).set(ylabel = '')

    x_index += 1

    if x_index == 4:

        y_index += 1

        x_index = 0



plt.subplots_adjust(hspace = 0.3)
X = pivot.drop(columns = ['Driving Distance', 'Launch Angle', 'Smash Factor', 'Spin Rate', 'Total Distance Efficiency'])

y = pivot['Driving Distance']



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state = 10)

#Normalizing data so that we all indendent variables will have equal weight in the model

scaler = preprocessing.StandardScaler()



#Fitting scaler model to my training data only. Fitting to train and test data assumes we know what our test data will be, and therefore is not an accurate test

scaler.fit(X_train)



#Transforming all data

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

#Creating a linear model using the training data

lm = linear_model.LinearRegression()

model = lm.fit(X_train,y_train)



#Evaluating the model against our test data

predictions = lm.predict(X_test)



#See how well the model performed

lm.score(X_test,y_test)

coef = pd.DataFrame(lm.coef_, X.columns, columns = ['Coefficient'])

coef
#Checking correlation matrix to see which features are heavily correlated

pivot.corr()

sns.heatmap(data=pivot.corr(), cmap = 'coolwarm')
#Creating variables for the predictor and target variables

X = pivot.drop(columns = ['Driving Distance', 'Launch Angle', 'Smash Factor', 'Spin Rate', 'Total Distance Efficiency'])

y = pivot['Driving Distance']



#Removing some additional predictor variables to reduce multicolinearity

X = X.drop(columns = 'Ball Speed') #'Club Head Speed' and 'Ball Speed' are likely colinear, as the club head speed directly affects the ball speed.



#'Carry Distance' in golf refers to the total distance the ball travels. Because this takes the height of the ball into account, we can drop "Distance to Apex", 

#"Apex Height", and 'Hang Time', which are also directly related to the height of the ball

X = X.drop(columns = ['Distance to Apex', 'Apex Height', 'Hang Time'])



#Carry Efficiency = Carry Distance / Club Head Speed. Carry Distance and Club Head Speed are part of our model, so we can remove this

#as it is essentially already accounted for.

X = X.drop(columns = 'Carry Efficiency')



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state = 10)



#Normalizing data so that we can gain more useful insights from our model coefficients

scaler = preprocessing.StandardScaler()



#Fitting scaler model to my training data only. Fitting to train and test data assumes we know what our test data will be, and therefore is not an accurate test

scaler.fit(X_train)



#Transforming all data

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



#Creating a linear model using the training data

lm = linear_model.LinearRegression()

model = lm.fit(X_train,y_train)



#Evaluating the model against our test data

predictions = lm.predict(X_test)



#See how well our model performed

lm.score(X_test,y_test)



coef = pd.DataFrame(lm.coef_, X.columns, columns = ['Coefficient'])

coef



print(coef)

print('Model score: ', lm.score(X_test,y_test))
sns.heatmap(data = X.corr(), cmap='coolwarm')
variables = pivot.drop(columns = 'Driving Distance').columns

results = pd.DataFrame(columns = ['Model score'])



for variable in variables:

    X = pivot[variable].values.reshape(-1,1)

    y = pivot['Driving Distance']



    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state = 10)



    scaler = preprocessing.StandardScaler()

    scaler.fit(X_train)



    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)



    lm = linear_model.LinearRegression()

    model = lm.fit(X_train,y_train)



    predictions = lm.predict(X_test)



    lm.score(X_test,y_test)



    coef = pd.DataFrame(lm.coef_)

    

    results.loc[variable] = lm.score(X_test,y_test)

    

results
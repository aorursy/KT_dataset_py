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
train1=pd.read_csv("/kaggle/input/eeg-dataset-of-slow-cortical-potentials/bsi_competition_ii_train1a.csv")

train2=pd.read_csv("/kaggle/input/eeg-dataset-of-slow-cortical-potentials/bsi_competition_ii_train1b.csv")
train1.describe()
train2.describe()
# Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')
train1.isnull().sum().sum()
train2.isnull().sum().sum()
fig = plt.figure(figsize=(30,2.5))

ax1 = fig.add_subplot(141)

sns.distplot(train1.loc[train1['0'] == 1]['1'], kde_kws={'label': 'positive'},ax=ax1);

sns.distplot(train1.loc[train1['0'] == 0]['1'], kde_kws={'label': 'negative'},ax=ax1);

ax2 = fig.add_subplot(142)

sns.distplot(train1.loc[train1['0'] == 1]['299'], kde_kws={'label': 'positive'},ax=ax2);

sns.distplot(train1.loc[train1['0'] == 0]['299'], kde_kws={'label': 'negative'},ax=ax2);

ax3 = fig.add_subplot(143)

sns.distplot(train1.loc[train1['0'] == 1]['597'], kde_kws={'label': 'positive'},ax=ax3);

sns.distplot(train1.loc[train1['0'] == 0]['597'], kde_kws={'label': 'negative'},ax=ax3);

ax4 = fig.add_subplot(144)

sns.distplot(train1.loc[train1['0'] == 1]['896'], kde_kws={'label': 'positive'},ax=ax4);

sns.distplot(train1.loc[train1['0'] == 0]['896'], kde_kws={'label': 'negative'},ax=ax4);

fig = plt.figure(figsize=(30,2.5))

ax1 = fig.add_subplot(141)

sns.distplot(train1.loc[train1['0'] == 1]['897'], kde_kws={'label': 'positive'},ax=ax1);

sns.distplot(train1.loc[train1['0'] == 0]['897'], kde_kws={'label': 'negative'},ax=ax1);

ax2 = fig.add_subplot(142)

sns.distplot(train1.loc[train1['0'] == 1]['1195'], kde_kws={'label': 'positive'},ax=ax2);

sns.distplot(train1.loc[train1['0'] == 0]['1195'], kde_kws={'label': 'negative'},ax=ax2);

ax3 = fig.add_subplot(143)

sns.distplot(train1.loc[train1['0'] == 1]['1493'], kde_kws={'label': 'positive'},ax=ax3);

sns.distplot(train1.loc[train1['0'] == 0]['1493'], kde_kws={'label': 'negative'},ax=ax3);

ax4 = fig.add_subplot(144)

sns.distplot(train1.loc[train1['0'] == 1]['1792'], kde_kws={'label': 'positive'},ax=ax4);

sns.distplot(train1.loc[train1['0'] == 0]['1792'], kde_kws={'label': 'negative'},ax=ax4);

fig = plt.figure(figsize=(30,2.5))

ax1 = fig.add_subplot(141)

sns.distplot(train1.loc[train1['0'] == 1]['1793'], kde_kws={'label': 'positive'},ax=ax1);

sns.distplot(train1.loc[train1['0'] == 0]['1793'], kde_kws={'label': 'negative'},ax=ax1);

ax2 = fig.add_subplot(142)

sns.distplot(train1.loc[train1['0'] == 1]['2091'], kde_kws={'label': 'positive'},ax=ax2);

sns.distplot(train1.loc[train1['0'] == 0]['2091'], kde_kws={'label': 'negative'},ax=ax2);

ax3 = fig.add_subplot(143)

sns.distplot(train1.loc[train1['0'] == 1]['2389'], kde_kws={'label': 'positive'},ax=ax3);

sns.distplot(train1.loc[train1['0'] == 0]['2389'], kde_kws={'label': 'negative'},ax=ax3);

ax4 = fig.add_subplot(144)

sns.distplot(train1.loc[train1['0'] == 1]['2688'], kde_kws={'label': 'positive'},ax=ax4);

sns.distplot(train1.loc[train1['0'] == 0]['2688'], kde_kws={'label': 'negative'},ax=ax4);

fig = plt.figure(figsize=(30,2.5))

ax1 = fig.add_subplot(141)

sns.distplot(train1.loc[train1['0'] == 1]['2689'], kde_kws={'label': 'positive'},ax=ax1);

sns.distplot(train1.loc[train1['0'] == 0]['2689'], kde_kws={'label': 'negative'},ax=ax1);

ax2 = fig.add_subplot(142)

sns.distplot(train1.loc[train1['0'] == 1]['2987'], kde_kws={'label': 'positive'},ax=ax2);

sns.distplot(train1.loc[train1['0'] == 0]['2987'], kde_kws={'label': 'negative'},ax=ax2);

ax3 = fig.add_subplot(143)

sns.distplot(train1.loc[train1['0'] == 1]['3285'], kde_kws={'label': 'positive'},ax=ax3);

sns.distplot(train1.loc[train1['0'] == 0]['3285'], kde_kws={'label': 'negative'},ax=ax3);

ax4 = fig.add_subplot(144)

sns.distplot(train1.loc[train1['0'] == 1]['3584'], kde_kws={'label': 'positive'},ax=ax4);

sns.distplot(train1.loc[train1['0'] == 0]['3584'], kde_kws={'label': 'negative'},ax=ax4);

fig = plt.figure(figsize=(30,2.5))

ax1 = fig.add_subplot(141)

sns.distplot(train1.loc[train1['0'] == 1]['3585'], kde_kws={'label': 'positive'},ax=ax1);

sns.distplot(train1.loc[train1['0'] == 0]['3585'], kde_kws={'label': 'negative'},ax=ax1);

ax2 = fig.add_subplot(142)

sns.distplot(train1.loc[train1['0'] == 1]['3883'], kde_kws={'label': 'positive'},ax=ax2);

sns.distplot(train1.loc[train1['0'] == 0]['3883'], kde_kws={'label': 'negative'},ax=ax2);

ax3 = fig.add_subplot(143)

sns.distplot(train1.loc[train1['0'] == 1]['4181'], kde_kws={'label': 'positive'},ax=ax3);

sns.distplot(train1.loc[train1['0'] == 0]['4181'], kde_kws={'label': 'negative'},ax=ax3);

ax4 = fig.add_subplot(144)

sns.distplot(train1.loc[train1['0'] == 1]['4480'], kde_kws={'label': 'positive'},ax=ax4);

sns.distplot(train1.loc[train1['0'] == 0]['4480'], kde_kws={'label': 'negative'},ax=ax4);

fig = plt.figure(figsize=(30,2.5))

ax1 = fig.add_subplot(141)

sns.distplot(train1.loc[train1['0'] == 1]['4481'], kde_kws={'label': 'positive'},ax=ax1);

sns.distplot(train1.loc[train1['0'] == 0]['4481'], kde_kws={'label': 'negative'},ax=ax1);

ax2 = fig.add_subplot(142)

sns.distplot(train1.loc[train1['0'] == 1]['4779'], kde_kws={'label': 'positive'},ax=ax2);

sns.distplot(train1.loc[train1['0'] == 0]['4779'], kde_kws={'label': 'negative'},ax=ax2);

ax3 = fig.add_subplot(143)

sns.distplot(train1.loc[train1['0'] == 1]['5077'], kde_kws={'label': 'positive'},ax=ax3);

sns.distplot(train1.loc[train1['0'] == 0]['5077'], kde_kws={'label': 'negative'},ax=ax3);

ax4 = fig.add_subplot(144)

sns.distplot(train1.loc[train1['0'] == 1]['5376'], kde_kws={'label': 'positive'},ax=ax4);

sns.distplot(train1.loc[train1['0'] == 0]['5376'], kde_kws={'label': 'negative'},ax=ax4);
from sklearn import datasets, linear_model, metrics 

from sklearn.model_selection import train_test_split 

from sklearn import metrics
train1_x=train1

train1_y=train1_x['0']
train1_x.drop('0',axis=1,inplace=True)
train1_x
train1_y
# load the boston dataset 

#boston = datasets.load_boston(return_X_y=False) 

# defining feature matrix(X) and response vector(y) 

X = train1_x

y = train1_y 

  

# splitting X and y into training and testing sets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 

                                                    random_state=1) 

  

# create linear regression object 

reg = linear_model.LinearRegression() 

  

# train the model using the training sets 

reg.fit(X_train, y_train) 

  

# regression coefficients 

print('Coefficients: \n', reg.coef_) 

  

# variance score: 1 means perfect prediction 

print('Variance score: {}'.format(reg.score(X_test, y_test))) 

# plot for residual error 

  

## setting plot style 

plt.style.use('fivethirtyeight') 

  

## plotting residual errors in training data 

plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 

            color = "green", s = 10, label = 'Train data') 

  

## plotting residual errors in test data 

plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 

            color = "blue", s = 10, label = 'Test data') 

  

## plotting line for zero residual error 

plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 

  

## plotting legend 

plt.legend(loc = 'upper right') 

  

## plot title 

plt.title("Residual errors") 

  

## function to show plot 

plt.show() 
X=X.astype(int)

y=y.astype(int)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test) 

print("Accuracy:",metrics.accuracy_score(y_test, y_pred.round()))
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object

clf = DecisionTreeClassifier(criterion="entropy")



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
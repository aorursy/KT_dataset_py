# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from statsmodels.graphics.correlation import plot_corr #correlation graph

from sklearn.model_selection import train_test_split #train and test split

from sklearn import preprocessing #label encoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#Import Data

data=pd.read_csv('/kaggle/input/iris/Iris.csv')
print(data.head())
#What is the distribution of the Species (y)

data.groupby('Species').size()
#Check for NULLs

data.isnull().sum().sum()



#there are no Nulls
# Use the 'hue' argument to provide a factor variable

sns.lmplot( x="SepalLengthCm", y="SepalWidthCm", data=data, fit_reg=False, hue='Species', legend=False)

 

# Move the legend to an empty part of the plot

plt.legend(loc='lower right')



plt.title("Iris Sepal Length vs Sepal Width")

#1) Correlation Matrix

corr = data[:-1].corr()

corr
#1) Correlation Matrix Heat Map

fig=plot_corr(corr,xnames=corr.columns)
#2) Another type of correlation matrix

sns.heatmap(data[data.columns[:-1]].corr(),annot=True)

fig=plt.gcf()

fig.set_size_inches(5,5)

plt.show()
#First convert species to a number

le = preprocessing.LabelEncoder()



for i in range(0,data.shape[1]):

    if data.dtypes[i]=='object':

        data[data.columns[i]] = le.fit_transform(data[data.columns[i]])

print(data.head(10))



data.groupby('Species').size()
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



print(X.head())

print(y.head())
#k-NN Model

# Instantiate learning model (k = 2)

clf = KNeighborsClassifier(2)



# Fitting the model

clf.fit(X_train,y_train)



# Predicting the Test set results

y_pred = clf.predict(X_test) 



print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test,y_pred))

print('k-NN Model Accuracy Score:', accuracy_score(y_test, y_pred))
#Log Model



classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



cm = confusion_matrix(y_test, y_pred)

print(cm)



# Command that outputs acccuracy

score = classifier.score(X_test, y_test)

print('Log Model Accuracy Score:', score)
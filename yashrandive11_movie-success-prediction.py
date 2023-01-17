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

data = pd.read_csv("../input/IMDB-Movie-Data.csv")
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

import random
data=data.dropna(axis=0, how='any')



#Data for Analysis

X = data[data.columns[6:32]]

Y=data.iloc[:,-1]



#Train and Test Splitting

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

scaler = StandardScaler()

X_train =scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)



#Model and Training

knn = KNeighborsClassifier(n_neighbors=5)

y_pred = knn.fit(X_train, Y_train).predict(X_test)



#Model Evaluation

conf_mat = confusion_matrix(Y_test,y_pred)

acc = accuracy_score(Y_test,y_pred)

precision = precision_score(Y_test,y_pred)

recall = recall_score(Y_test,y_pred)

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Print Results

print('Confusion Matrix :')

print(conf_mat)

print('\nAccuracy is :')

print(acc)

print('\nPrecision is :')

print(precision)

print('\nRecall is: ')

print(recall)
## Importing Additional Required Libraries for Naive Bayes Algorithm

from sklearn.naive_bayes import GaussianNB



data=data.dropna(axis=0, how='any')



#Data for Analysis

X = data[data.columns[6:32]]

Y=data.iloc[:,-1]



#Train and Test Splitting

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

scaler = StandardScaler()

X_train =scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)



#Model and Training

gnb = GaussianNB()

y_pred = gnb.fit(X_train,Y_train).predict(X_test)



#Model Evaluation

conf_mat = confusion_matrix(Y_test,y_pred)

acc = accuracy_score(Y_test,y_pred)

precision = precision_score(Y_test,y_pred)

recall = recall_score(Y_test,y_pred)

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Print Results

print('Confusion Matrix :')

print(conf_mat)

print('\nAccuracy :')

print(acc)

print('\nPrecision is :')

print(precision)

print('\nRecall is: ')

print(recall)
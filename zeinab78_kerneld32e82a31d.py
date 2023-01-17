# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import sys

from pandas import Series

import pandas as pd

import numpy as np

import traceback

import time

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/VitalSigns.csv")

data.head()
print('Shape of the data set: ' + str(data.shape))
Temp = pd.DataFrame(data.isnull().sum())

Temp.columns = ['Environmental_Temperature']

print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['Environmental_Temperature'] > 0])) )
Temp = pd.DataFrame(data.isnull().sum())

Temp.columns = ['PRbpm']

print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['PRbpm'] > 0])) )
Temp = pd.DataFrame(data.isnull().sum())

Temp.columns = ['Airflow']

print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['Airflow'] > 0])) )
plt.title('Environmental Temperature')

plt.xlabel('Time(s)')

plt.ylabel('Temperature')

aX=data['Environmental_Temperature']

Time=data['Second']

plt.plot(aX,  color='red')

plt.legend() 
plt.title('Pulse Rate')

plt.xlabel('Time(s)')

plt.ylabel('Pulse Rate')

aX=data['PRbpm']

Time=data['Second']

plt.plot(aX,  color='red')



plt.title('Air Flow')

plt.xlabel('Time(s)')

plt.ylabel('Air flows')

aX=data['Airflow']

Time=data['Second']

plt.plot(aX,  color='red')
X = data.drop('class',axis=1).values

y = data['class'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)
neighbors = np.arange(1,9)

train_accuracy =np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



for i,k in enumerate(neighbors):

    #Setup a knn classifier with k neighbors

    knn = KNeighborsClassifier(n_neighbors=k)

    

    #Fit the model

    knn.fit(X_train, y_train)

    

    #Compute accuracy on the training set

    train_accuracy[i] = knn.score(X_train, y_train)

    

    #Compute accuracy on the test set

    test_accuracy[i] = knn.score(X_test, y_test)
plt.title('k-NN Varying number of neighbors')

plt.plot(neighbors, test_accuracy, label='Testing Accuracy')

plt.plot(neighbors, train_accuracy, label='Training accuracy')

plt.legend()

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')

plt.show()
#Setup a knn classifier with k neighbors

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
#let us get the predictions using the classifier we had fit above

y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
print(classification_report(y_test,y_pred))
data.dropna(inplace=True)

data.drop_duplicates(inplace=True)

sns.pairplot(data);
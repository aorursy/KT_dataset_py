# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# The following notebook uses Support Vector Machines on the famous Iris dataset.

# This dataset was introduced by the British statistician and biologist Sir Ronald Fisher 

# in his 1936 paper The use of multiple measurements in taxonomic problems



# This dataset is openly available at UCI Machine Learning Repository
#The iris dataset contains measurements for 150 iris flowers from three different species.



#The three classes in the Iris dataset:



#    Iris-setosa (n=50)

#    Iris-versicolor (n=50)

#    Iris-virginica (n=50)



# The four features of the Iris dataset:



#    sepal length in cm

#    sepal width in cm

#    petal length in cm

#    petal width in cm



## Get the data



#**Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') **
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Import the dataset using Seaborn library

iris=pd.read_csv('../input/IRIS.csv')
# Checking the dataset

iris.head()
# Creating a pairplot to visualize the similarities and especially difference between the species

sns.pairplot(data=iris, hue='species', palette='Set2')
from sklearn.model_selection import train_test_split
# Separating the independent variables from dependent variables

import hashlib 



def myHash1(XX):

    for i in range(0, len(XX)):

        temp = int(hashlib.sha256(str(XX[i]).encode('utf-8')).hexdigest(), 16)

        #temp = int(hashlib.md5(str(XX[i])).hexdigest(), 16)

        XX[i] = (temp % (10**6))

        #XX[i] = temp

    return XX



#print(iris)

x=iris.iloc[:,:-1]

print("Normal Dataset")

print(x)



x1=iris.iloc[:,:-1]

x1['petal_width'] = myHash1(x1['petal_width'])

x1['petal_length'] = myHash1(x1['petal_length'])

x1['sepal_width'] = myHash1(x1['sepal_width'])

x1['sepal_length'] = myHash1(x1['sepal_length'])

print("Dataset with Hashed Features")

print(x1)



y=iris.iloc[:,4]

x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.30)

x_train1,x_test1, y_train1, y_test1=train_test_split(x1,y,test_size=0.30)
from sklearn.svm import SVC

model=SVC()

model1=SVC()
model.fit(x_train, y_train)

model1.fit(x_train1, y_train1)
pred=model.predict(x_test)

pred_train=model.predict(x_train)



pred1=model1.predict(x_test1)

pred_train1=model1.predict(x_train1)
# Importing the classification report and confusion matrix

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))

print(confusion_matrix(y_test1,pred1))
print("Normal Training")

print("train error:")

print(classification_report(y_train, pred_train))

print('test error:')

print(classification_report(y_test, pred))

print()

print("Training with Hashed Features")

print("train error:")

print(classification_report(y_train1, pred_train1))

print('test error:')

print(classification_report(y_test1, pred1))

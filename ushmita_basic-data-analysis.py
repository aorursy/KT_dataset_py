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
#Data input

print(filename)

import pandas as pd

data=pd.read_csv('/kaggle/input/diabetes.csv')
#installing library

import numpy as np

import seaborn as sns

from scipy.stats import pearsonr

%matplotlib inline

from matplotlib import pyplot as plt
#Data stucture information

data.info()
#Descriptive Statistics

data.describe()
#To cheak null in column

data.isna()
#To cheak in any given colunm there is null or not.This is more effective as we can see that no null value is present in any column

data.isnull().any()
#To see the number of diabetes cases.Here 0=no diabates and 1= having diabates.

plt.subplots(figsize=(7,7))

plo=data['Outcome'].value_counts().sort_values().plot.barh()
#Correlation among different metrics has been tested.

sns.heatmap(data.corr(),annot=True,cmap="Set3")
preg=sns.kdeplot(data['Pregnancies'])
glu=sns.kdeplot(data['Glucose'])
bloodpr=sns.kdeplot(data['BloodPressure'])
SkinThi=sns.kdeplot(data['SkinThickness'])
sns.kdeplot(data['Insulin'])
sns.kdeplot(data['BMI'])
sns.kdeplot(data['Age'])
fig_dims = (14, 14)

fig, ax = plt.subplots(figsize=fig_dims)

sns.countplot('Age',hue='Outcome',data=data)
group=data.groupby('Outcome')
group.max()
group.mean()
type(data)
data.info()
Y=data['Outcome']
data.drop('Outcome',axis=1,inplace=True)
print(data)
#Here k=1

from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors=1)

print(knn)
knn.fit(data,Y)
data.describe()
#Predicting by giving the new feature values.Here I have given the mean of each feature.

predi= knn.predict([[3.8,120,69,20,79,31,0.47,33]])
print(predi)
#Predicting by giving the new feature values.Here I have given the mean of each feature.

data.median()
pred= knn.predict([[3.8,120,69,20,79,31,0.47,33],[3,117,72,23,30,32,0.32,29]])
print(pred)
#Here k=5

knn5= KNeighborsClassifier(n_neighbors=5)

print (knn5)
knn5.fit(data,Y)
predi5= knn5.predict([[3.8,120,69,20,79,31,0.47,33]])
print(predi5)
pred5= knn5.predict([[3.8,120,69,20,79,31,0.47,33],[3,117,72,23,30,32,0.32,29]])
print(pred5)
from sklearn.linear_model import LogisticRegression

logc= LogisticRegression()

logc.fit(data,Y)
prelog= logc.predict([[3.8,120,69,20,79,31,0.47,33],[3,117,72,23,30,32,0.32,29]])

print(prelog)
#To divide the date set into Train and Test

from sklearn.model_selection import train_test_split

X_train, X_test ,y_train, y_test =train_test_split(data,Y,test_size=0.4,random_state= 4)
print(X_train.shape)

print(X_test.shape)
logc.fit(X_train,y_train)

pr = logc.predict(X_test)

print(pr)
#This is the accuracy from logistic regresion.That means there is 79% accuracy in result.

from sklearn import metrics

print(metrics.accuracy_score(y_test, pr))
#Accurecy from knn where k=1

knn.fit(X_train,y_train)

prk= knn.predict(X_test)

print(metrics.accuracy_score(y_test, prk))
#Acuracy from knn when k=5

knn5.fit(X_train,y_train)

prk5= knn5.predict(X_test)

print(metrics.accuracy_score(y_test, prk5))
#Here we are trying to apply cross validation where we finalizing your optimum algorithum.Here cv is the number of itteration it should perform

from sklearn.model_selection import *

knn5= KNeighborsClassifier(n_neighbors=5)

score= cross_val_score(knn5, data,Y,cv=10 , scoring='accuracy')

print(score.mean())
#To check accuracy from knn across different k value

k_range= range(1,45)

k_scores =[]

for k in k_range:

    knn= KNeighborsClassifier(n_neighbors = k)

    scores=cross_val_score(knn,data,Y, cv=10, scoring='accuracy')

    k_scores.append(scores.mean())
print(k_scores)
import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(k_range,k_scores)



plt.xlabel('k values in knn')

plt.ylabel('mean accuracy score')
#To create confusion metrics as it helps to undustand the acurracy more in logistic regression.

from sklearn import metrics

cnf_matrix=metrics.confusion_matrix(y_test,pr)

cnf_matrix
class_names=[0,1]

plt.subplot()

tick_marks=np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)

sns.heatmap(pd.DataFrame(cnf_matrix),annot=True)

plt.title('confusion matrix')

plt.ylabel('Actual')

plt.xlabel('Predicted')
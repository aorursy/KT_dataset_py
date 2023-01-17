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
#import libraries



import seaborn as sns

import matplotlib.pyplot as plt

#for jupyter notebook we use this line

%matplotlib inline                                    

sns.set_style('whitegrid')
# Read train data

Titanicdata=pd.read_csv("../input/titanic/train.csv")

Titanictest=pd.read_csv("../input/titanic/test.csv")
#Check the 10 samples for data

Titanicdata.head(10)
#check simple information like  columns names ,  columns datatypes and null values

Titanicdata.info()
#check summary of numerical data  such as count , mean , max , min  and standard deviation.

Titanicdata.describe()
#check numbers of rows(samples) and columns(features)

Titanicdata.shape
#check count of values for each features

Titanicdata.count()
#Check total missing values in each feature of train data

Titanicdata.isnull().sum()
#Check total missing values in each feature of test data

Titanictest.isnull().sum()
#Delete PassengerId,Cabin, Ticket  useless features.

#Cabin has  a lot of missing values

Titanicdata.drop(["PassengerId","Cabin","Ticket"],axis = 1, inplace = True)

Titanictest.drop(["PassengerId","Cabin","Ticket"],axis = 1, inplace = True)

Titanicdata["Sex"].value_counts()
groubBySurvived=Titanicdata.groupby("Survived").size()

no_Survivors=groubBySurvived[1]

no_Deaths=groubBySurvived[0]

print("Numbers of People Survivers: {} \nNumbers of People Deaths: {}".format(no_Survivors,no_Deaths))
class_sex_grouping = Titanicdata.groupby(['Pclass','Sex']).count()

class_sex_grouping
class_sex_grouping['Survived'].plot.pie()
Embarked_sex_grouping = Titanicdata.groupby(['Embarked','Sex',]).count()

Embarked_sex_grouping
Embarked_sex_grouping['Pclass'].plot.bar()
sns.pairplot(Titanicdata)
sns.countplot(x="Sex",data=Titanicdata)
sns.barplot('Embarked', 'Survived', data=Titanicdata)
sns.barplot('Pclass', 'Survived', data=Titanicdata)
Titanicdata.Sex=pd.Categorical(Titanicdata.Sex,['male','female'],ordered=True)

Titanicdata.Sex=Titanicdata.Sex.cat.codes
Titanicdata.Embarked=pd.Categorical(Titanicdata.Embarked,['S','C','Q'],ordered=True)

Titanicdata.Embarked=Titanicdata.Embarked.cat.codes
Titanictest.isnull().sum()
#fill Age feature with  measure of mean or median

Titanicdata["Age"].fillna(Titanicdata["Age"].mean(), inplace = True)

Titanictest["Age"].fillna(Titanictest["Age"].mean(), inplace = True) 
#Titanicdata=Titanicdata.replace("",np.nan)

Titanictest=Titanicdata.replace("",np.nan)
#fill Fare feature with  measure of mode

Titanicdata["Fare"].fillna(Titanicdata["Fare"].mode(), inplace = True)

Titanictest["Fare"].dropna()
#fill Embarked feature with  measure of mode, Embarked has 2 missing values only.



Titanicdata["Embarked"].fillna(Titanicdata["Embarked"].mode(), inplace = True)

Titanicdata['Sex']=Titanicdata['Sex'].replace('female',0)

Titanicdata['Sex']=Titanicdata['Sex'].replace('male',1)

Titanictest['Sex']=Titanictest['Sex'].replace('female',0)

Titanictest['Sex']=Titanictest['Sex'].replace('male',1)
Titanicdata['Embarked']=Titanicdata['Embarked'].replace('S',0)

Titanicdata['Embarked']=Titanicdata['Embarked'].replace('C',1)

Titanicdata['Embarked']=Titanicdata['Embarked'].replace('Q',2)

Titanictest['Embarked']=Titanictest['Embarked'].replace('S',0)

Titanictest['Embarked']=Titanictest['Embarked'].replace('C',1)

Titanictest['Embarked']=Titanictest['Embarked'].replace('Q',2)

Cols=["Age","Embarked","Parch","Pclass","Sex","Fare","SibSp"]

X_train=Titanicdata[Cols]

X_test=Titanictest[Cols]

y=Titanicdata["Survived"]

y_test=Titanictest["Survived"]

from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()

y_train= labelencoder_y.fit_transform(y)

y_test=labelencoder_y.fit_transform(y_test)

Titanictest.isnull().sum()


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(n_neighbors =13, metric = 'minkowski', p = 2)

knc.fit(X_train, y_train)

y_pred = knc.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

#print(cm)

accuracy= knc.score(X_test, y_test)

print(accuracy)
#Support Vector classifier

from sklearn.svm import SVC

svc = SVC(kernel = 'rbf', random_state = 42)

svc.fit(X_train, y_train)



# Predicting the Test set results

y_pred = svc.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy= svc.score(X_test, y_test)

print(accuracy)
#Support Vector classifier

from sklearn.svm import SVC

svc = SVC(kernel = 'poly', random_state = 42)

svc.fit(X_train, y_train)



# Predicting the Test set results

y_pred = svc.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy= svc.score(X_test, y_test)

print(accuracy)
#DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=0)

dtc.fit(X_train,y_train)

y_pred=dtc.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

#print(cm)

accuracy= dtc.score(X_test, y_test)

print(accuracy)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=0)

rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

#print(cm)

accuracy= dtc.score(X_test, y_test)

print(accuracy)

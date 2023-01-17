# Importing necessary Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Reading the dataset using pandas built-in function 'read_csv'

ds = pd.read_csv('/kaggle/input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
ds
# GETTING INFO OF dataset
ds.info()
# Checking whether the dataset have 'nan' values or not

ds.isnull().sum()
# Encoding the Class/Target labels from object type to int..!

from sklearn.preprocessing import LabelEncoder

Class = LabelEncoder()
ds['Class_n'] = Class.fit_transform(ds['class'])

ds.drop('class', axis=1, inplace=True)
ds.head()
# checking the total values of target labels 
ds['Class_n'].value_counts()
galaxy_ds = ds[ds['Class_n'] == 0]                      # have all values that have class/target label as 'Galaxy'
qso_ds = ds[ds['Class_n'] == 1]                         # have all values that have class/target label as 'QSO'
star_ds = ds[ds['Class_n'] == 2]                        # have all values that have class/target label as 'Star'

galaxy_ds = galaxy_ds.sample(qso_ds.shape[0])           # getting any 850 random values from 'galaxy_ds' dataset
star_ds = star_ds.sample(qso_ds.shape[0])               # getting any 850 random values from 'star_ds' dataset

# now we have to append these three datasets
df = qso_ds.append(galaxy_ds, ignore_index=True)        
ds = star_ds.append(df, ignore_index=True)
ds.shape
ds['Class_n'].value_counts()

# now the dataset is balanced 
# spliting the dataset into train and test

from sklearn.model_selection import train_test_split

x = ds.drop('Class_n', axis=1)
y = ds['Class_n']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.24)

x_train.shape , x_test.shape
from sklearn.feature_selection import VarianceThreshold

filter = VarianceThreshold()

x_train = filter.fit_transform(x_train)
x_test = filter.fit_transform(x_test)

x_train.shape , x_test.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# converting the labels series into numpy array because it is more faster..!

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
# Importing Machine Learning Algorithm 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
# Decision tree training with accuracy result
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt.score(x_test,y_test)
# Random Forest training with accuracy result
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf.score(x_test,y_test)
# XGBClassifier training with accuracy result
xgb = XGBClassifier()
xgb.fit(x_train,y_train)
xgb.score(x_test,y_test)
# Naive byes training with accuracy result
nb = GaussianNB()
nb.fit(x_train,y_train)
nb.score(x_test,y_test)
# KNeighborsClassifier training with accuracy result
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
knn.score(x_test,y_test)
# SVM training with accuracy result
svm = SVC()
svm.fit(x_train,y_train)
svm.score(x_test,y_test)
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = xgb.predict(x_test)

acs = accuracy_score(y_test, y_pred)
print('Accuracy Score of XGB Classifier: ', acs)
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n',cm)
plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('truth')
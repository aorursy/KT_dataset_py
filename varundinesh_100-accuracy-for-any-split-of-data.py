# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/mushrooms.csv')
#Checking for duplicates
tot=len(set(data.index))
last=data.shape[0]-tot
last
#Checking for null values
data.isnull().sum()

#checking the shape of dataset
data.shape
#Lets see how the target variable is balanced
print(data['class'].value_counts())
sns.countplot(x='class', data=data)
plt.show()
#Looking for categorical data
cat=data.select_dtypes(include=['object']).columns
cat
#detailed view of each columns
for c in cat:
    print(c)
    print("-"*50)
    print(data[c].value_counts())
    sns.countplot(x=c, data=data)
    plt.show()
    print("-"*50)
#we will remove what all we think not important or less contribution to target
data['cap-shape']=data[data['cap-shape']!='c']
data.dropna(inplace=True)
data.shape
data['cap-surface']=data[data['cap-surface']!='g']
data.dropna(inplace=True)
data.shape
data.drop('veil-type',axis=1,inplace=True)
cat=data.select_dtypes(include='object').columns
cat
#lets convert categorical data to numerical
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in cat:
    data[i]=le.fit_transform(data[i])
    
f,ax = plt.subplots(figsize=(20, 15))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
#lets do some feature engineering for fun
data['f-engineer']=((data['gill-size']+5)*(data['population']+5)*(1/((data['gill-color']+5)*(data['bruises']+5)*(data['ring-type']+5))))
f,ax = plt.subplots(figsize=(20, 15))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
X = data.iloc[:,1:]
X = X.values
y = data['class'].values
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
algo = {'LR': LogisticRegression(), 
        'DT':DecisionTreeClassifier(), 
        'RFC':RandomForestClassifier(n_estimators=100), 
        'SVM':SVC(gamma=0.01),
        'KNN':KNeighborsClassifier(n_neighbors=10)
       }

for k, v in algo.items():
    model = v
    model.fit(X_train, y_train)
    print('Acurracy of ' + k + ' is {0:.2f}'.format(model.score(X_test, y_test)*100)+'%')
#yes we have accuracy of 100%
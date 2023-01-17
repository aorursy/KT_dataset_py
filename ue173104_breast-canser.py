# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from numpy.core.umath_tests import inner1d
%matplotlib inline
import seaborn as sea
import warnings
warnings.filterwarnings('ignore')
train=pd.read_csv('../input/data.csv')
train.head()
train['diagnosis'].value_counts()
train.isnull().sum()*100/train.shape[0]
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
train['diagnosis']=label.fit_transform(train['diagnosis'])
train.head()
train.drop('Unnamed: 32',axis=1,inplace=True)
target=train['diagnosis']
train.drop('diagnosis',axis=1,inplace=True)
target.value_counts().plot.bar()
train.drop('id',axis=1,inplace=True)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,target,test_size=0.25,random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
classi=[LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(),GaussianNB(),KNeighborsClassifier(),MLPClassifier(),XGBClassifier()]
from sklearn.model_selection import cross_val_score
i=0
for x in classi:
    cv=cross_val_score(x,train,target,cv=10)
    print(i,cv.mean())
    i=i+1
classi[6]
classifier=XGBClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
(89+51)/x_test.shape[0]



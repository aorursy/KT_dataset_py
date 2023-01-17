import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC
data=pd.read_csv('../input/fish-market/Fish.csv')
data.head()
data.info()
data.describe()
X=data.drop(['Species'],axis=1)
##to make multi calss model so convert fish name to int

y= data['Species'].map({'Perch': 0, 'Bream': 1,'Roach': 2,'Pike': 3,'Smelt': 4,'Parkki':5,'Whitefish':6})
X_std=StandardScaler().fit(X).transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_std,y,random_state=0)
LinearSVC_result=[]

def get(_X,_y,_X_test,_y_test):

    for i in {1,10,50,100,1000}:

        model=LinearSVC(C=i,random_state=0)

        model.fit(_X,_y)

        LinearSVC_result.append(model.score(_X_test,_y_test))

    return LinearSVC_result
get(X_train,y_train,X_test,y_test)
LogisticRegression_result=[]

def get(_X,_y,_X_test,_y_test):

    for i in {1,10,50,100,1000}:

        model=LogisticRegression(C=i,random_state=0)

        model.fit(_X,_y)

        LogisticRegression_result.append(model.score(_X_test,_y_test))

    return LogisticRegression_result
get(X_train,y_train,X_test,y_test)
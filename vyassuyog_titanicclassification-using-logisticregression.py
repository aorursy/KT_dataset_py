# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.optim as opt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data=pd.read_csv("../input/titanic/train.csv")
data

data.describe()
data=data.fillna(0)
X_train=data.drop(["Survived"],axis=1)
X_train

Y_train=data["Survived"]
Y_train

Cabin_mean=data.groupby("Cabin")["Survived"].mean()

X_train.loc[:,"Cabin_mean"]=X_train.loc[:,"Cabin"].map(Cabin_mean)
embarked_mean=data.groupby("Embarked")["Survived"].mean()
embarked_mean
X_train.loc[:,"Embarked_mean"]=X_train.loc[:,"Embarked"].map(embarked_mean)
X_train=X_train.drop('PassengerId',axis=1)
X_train
X_train=X_train.drop(["Name","Ticket","Embarked","Cabin"],axis=1)

label=LabelEncoder()
sex_label=pd.DataFrame(label.fit_transform(X_train["Sex"]))
X_train=X_train.join(sex_label)

X_train=X_train.drop(["Sex"],axis=1)
X_train

plt.scatter(X_train.loc[:,"Cabin_mean"],Y_train)
X_tra=X_train.copy()
X_tra
Y_tra=Y_train.copy()
Y_tra
X_tra
X_traint,X_valt,Y_traint,Y_valt=train_test_split(X_tra,Y_tra,stratify=Y_tra,test_size=0.2,random_state=0)
print(X_traint.shape)
print(X_valt.shape)
print(Y_traint.shape)
print(Y_valt.shape)
X_traint
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

logreg = LogisticRegression()


logreg.fit(X_traint, Y_traint)
y_pred = logreg.predict(X_valt)
acc_logreg = round(accuracy_score(y_pred, Y_valt) * 100, 2)
print(acc_logreg)

Submit_val=pd.DataFrame(X_valt)
Submit_val=Submit_val.join(Y_valt)
Submit_val.reset_index(level=0,inplace=True)
Submit_val.rename(columns = {0:'Sex'}, inplace = True) 

y_pred=pd.DataFrame(y_pred,index=None)
Submit_val=pd.concat([Submit_val,y_pred],axis=1)
Submit_val.rename(columns = {0:'Y_pred'}, inplace = True) 
Submit_val


test_data=pd.read_csv('../input/titanic/test.csv')
test_data
test_data=test_data.fillna(0)
test_data.loc[:,"Cabin_mean"]=test_data.loc[:,"Cabin"].map(Cabin_mean)
embarked_mean=data.groupby("Embarked")["Survived"].mean()
embarked_mean
test_data.loc[:,"Embarked_mean"]=test_data.loc[:,"Embarked"].map(embarked_mean)
test_data=test_data.drop(["PassengerId","Name","Ticket","Embarked","Cabin"],axis=1)
test_data
label=LabelEncoder()
sex_label=pd.DataFrame(label.fit_transform(test_data["Sex"]))
test_data=test_data.join(sex_label)
test_data=test_data.drop(["Sex"],axis=1)
test_data
test_data.rename(columns={0:"Sex"},inplace=True)
test_data.fillna(0,inplace=True)
test_data
y_pred = logreg.predict(test_data)
y_pred=pd.DataFrame(y_pred)
test_data=pd.concat([test_data,y_pred],axis=1)
test_data
test_data.rename(columns={0:"Preedicted_Survived"},inplace=True)
test_data
test_data.to_csv("test_submission.csv")
test_data.to_csv("test_submission.csv")
Submit.to_csv("train_val_submission.csv")

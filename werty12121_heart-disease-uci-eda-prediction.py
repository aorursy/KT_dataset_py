import numpy as np

np.random.seed(1)



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import copy

import os

print(os.listdir("../input"))



# optional

# sns.set(style="whitegrid")
data=pd.read_csv("../input/heart.csv")

data.head()
sns.distplot(data.target)
cmap = sns.diverging_palette(250, 15, s=75, l=40,n=9, center="dark")

f, ax = plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(),square=True,annot=True,vmin=-1,vmax=1, center=0,cmap=cmap)
def draw_data_analysis(data,x,y,ws=1):

    size=(max(10,len(pd.unique(data[x]))/ws),10)

    f, ax = plt.subplots(figsize=size)    

    sns.countplot(data[x])

    plt.title("Countplot "+x)

    plt.show()

    

    f, ax = plt.subplots(figsize=size)

    sns.barplot(x=x,y=y,data=data)

    plt.title("Barplot "+x)

    plt.show()
for column in data.columns.drop('target'):

    draw_data_analysis(data,column,'target',2)
data.isna().sum()
data_prev=copy.copy(data)

data.age=pd.cut(data.age,15)

data.trestbps=pd.cut(data.trestbps,5)

data.chol=pd.cut(data.chol,20)

data.thalach=pd.cut(data.thalach,10)

data.oldpeak=pd.cut(data.oldpeak,10)

# data.thal=data.thal.astype('category')
for column in data.columns.drop('target'):

    draw_data_analysis(data,column,'target',1/2)
f, ax = plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(),square=True,annot=True,vmin=-1,vmax=1, center=0,cmap=cmap)
data.info()
data.describe()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split

import xgboost as xgb

from sklearn.metrics import accuracy_score,auc

from sklearn.ensemble import RandomForestClassifier



y=data.target

data=data.drop(['target'],axis=1)

num_feat=data.dtypes[data.dtypes!='category'].index



sc=StandardScaler()

data[num_feat]=sc.fit_transform(data[num_feat])
data.describe()
kfolds = KFold(n_splits=10, shuffle=True,random_state=1)

def acc_cv(model):

    acc= cross_val_score(model, data.values, y.values, scoring="roc_auc", cv = kfolds.get_n_splits(data.values))

    return acc


data=pd.get_dummies(data)



X_train, X_test, y_train, y_test = train_test_split(data.values, y.values, test_size=0.4,random_state=100)

rf=RandomForestClassifier(30,random_state=1)

rf.fit(X_train,y_train)

print(acc_cv(rf).mean())



xgbm=xgb.XGBClassifier(objective='binary:hinge',random_state=1)

xgbm.fit(X_train,y_train)

print(acc_cv(xgbm).mean())
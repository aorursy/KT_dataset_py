import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing as pp

%matplotlib inline
df=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")
df_new  = df.drop(['id'],axis=1)
y=df_new['rating']
X = df_new.drop(['rating'],axis=1)
numX = X.drop(['type'],axis=1)
CatX = X['type']
numX.fillna(numX.mean(),inplace = True)
normalizernx = pp.Normalizer();
numXnorm = normalizernx.fit_transform(numX)
X_train = pd.DataFrame(numXnorm,columns=list(numX.columns))
X_train["type"]=CatX 
X_train["type"]=X["type"].map({"new":1,"old":0})
X_train
X_test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
X_test['type']=X_test['type'].map({'new':1,'old':0})
catXtest=X_test['type']
X_test=X_test.drop(['type'],axis=1)

X_test = X_test.drop(['id'],axis=1)
X_test.fillna(X_test.mean(),inplace = True)
norm2=pp.Normalizer()
X_test
X_test=norm2.fit_transform(X_test)
X_test = pd.DataFrame(X_test,columns=list(numX.columns))
X_test
X_test['type']=catXtest
X_test
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



# Initialize and train

clf1 = DecisionTreeClassifier(max_depth = 10)

clf2 = RandomForestClassifier(n_estimators=50,max_depth=15)
clf2.fit(X_train,y)
y_pred = clf2.predict(X_test)
new=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
dfpref = pd.DataFrame({'id':new['id'],"rating":y_pred})
dfpref.head()
dfpref.to_csv ("sub3.csv", header=True,index=False)
dfpref.head()
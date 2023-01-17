import pandas as pd

import numpy as np
df=pd.read_csv('../input/glass.csv')
df.head()
from sklearn.utils import shuffle

df = shuffle(df)
X=df.ix[:,:-1]

Y=df.ix[:,-1]
xtest=X.ix[:100,]

ytest=Y.ix[:100,]

xtrain=X.ix[100:,]

ytrain=Y.ix[100:,]
from sklearn import ensemble
RMClassifier=ensemble.RandomForestClassifier()
RMClassifier.fit(xtrain,ytrain)
RMClassifier.score(xtest,ytest)
type(RMClassifier.score(xtest,ytest))
from sklearn.neighbors import KNeighborsClassifier
KNClassifier=KNeighborsClassifier()

KNClassifier.fit(xtrain,ytrain)
KNClassifier.score(xtest,ytest)
print(df['Type'].value_counts().sort_values(ascending=False)) # six categories 

print()
from sklearn import tree

clf=tree.DecisionTreeClassifier()
clf.fit(xtrain,ytrain)
clf.score(xtest,ytest)
import seaborn as sns

import matplotlib.pyplot as plt
df1=df.corr()

#seperating the ideal coefficient

df1=df1[df1<1]

sns.heatmap(df1,annot=True)
df.groupby(by='Type').mean()
sns.countplot(data=df,x='Type')
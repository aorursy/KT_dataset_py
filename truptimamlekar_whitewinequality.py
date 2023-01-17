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
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

#from sklearn.model_selection

import numpy as np

from collections import Counter

import collections

import os

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score

from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings("ignore")
df2=pd.read_csv('../input/wine-quality/winequalityN.csv')

df2.head(5)
df2=df2.drop(['type'],axis=1)
df2.describe()
df2.info ()
df2.isnull().values.any()
df2=df2.dropna()
plt.figure(figsize=(10, 6))

sns.countplot(df2["quality"], palette="muted")

df2["quality"].value_counts()
df2.plot.box (sym='k.',figsize=(15,10))

plt.show ()
quality = df2["quality"].values

category = []

for num in quality:

    if num < 5.5:

        category.append("Low")

    else:

        category.append("High")

[(i, category.count(i)) for i in set(category)]

plt.figure(figsize=(10, 6))

sns.countplot(category, palette="muted")
f,ax=plt.subplots(1,1,figsize=(25,6))

ax = sns.scatterplot(x="volatile acidity", y="quality",color = "red",data=df2)

ax = sns.scatterplot(x="citric acid", y="quality",color = "green",data=df2)

ax = sns.scatterplot(x="chlorides", y="quality",color = "blue",data=df2)
f,ax=plt.subplots(1,1,figsize=(25,6))

sns.kdeplot(df2.loc[(df2['quality']==4), 'alcohol'], color='b', shade=True, Label='4')

sns.kdeplot(df2.loc[(df2['quality']==5), 'alcohol'], color='g', shade=True, Label='5')

sns.kdeplot(df2.loc[(df2['quality']==6), 'alcohol'], color='r', shade=True, Label='6')

sns.kdeplot(df2.loc[(df2['quality']==7), 'alcohol'], color='y', shade=True, Label='7')

plt.xlabel('Alcohol') 

plt.ylabel('Probability Density') 
f,ax=plt.subplots(2,2,figsize=(25,15))

sns.violinplot(x="quality", y="fixed acidity",ax=ax[0][0],data=df2, palette="muted")

sns.violinplot(x="quality", y="volatile acidity",data=df2,ax=ax[0][1], palette="muted")

sns.violinplot(x="quality", y="citric acid",ax=ax[1][0],data=df2, palette="muted")

sns.violinplot(x="quality", y="residual sugar",ax=ax[1][1],data=df2, palette="muted")
plt.figure(figsize=(12, 6))

sns.heatmap(df2.corr(), annot=True)
import matplotlib.pyplot as plt

import seaborn as sns

df2.hist (bins=10,figsize=(20,20))

plt.show ()
sns.pairplot(df2)
features=df2.columns

features=features.drop(['quality'])

x=df2 [features]

y=df2 ['quality']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)
seed=7

models = []

models.append(('RF',RandomForestClassifier()))

models.append(('DTC',DecisionTreeClassifier()))

models.append(('SVM',SVC()))

models.append(('LR',LogisticRegression()))

models.append(('NB',GaussianNB()))

# Evaluating each models in turn

results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(n_splits=10,random_state=seed)

    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())

    print(msg)
logistic = LogisticRegression()

logistic.fit(x_train,y_train)

y_pred=logistic.predict(x_test)

print(classification_report(y_test,y_pred))

accuracy1=logistic.score(x_test,y_test)

print (accuracy1*100,'%')
classifier=SVC()

classifier.fit(x_train,y_train)

svm_predict=classifier.predict(x_test)

print(classification_report(y_test,svm_predict))

accuracy2=classifier.score(x_test,y_test)

print(accuracy2*100,'%')
ran_class=RandomForestClassifier()

ran_class.fit(x_train,y_train)

ran_predict=ran_class.predict(x_test)

print(classification_report(y_test,ran_predict))

accuracy3=ran_class.score(x_test,y_test)

print(accuracy3*100,'%')
nvclass=GaussianNB()

nvclass.fit(x_train,y_train)

y_pr=nvclass.predict(x_test)

print(classification_report(y_test,y_pr))

accuracy4=nvclass.score(x_test,y_test)

print(accuracy4*100,'%')
model1 = DecisionTreeClassifier(random_state=1)

model1.fit(x_train, y_train)

y_pred1 = model1.predict(x_test)

print(classification_report(y_test, y_pred1))

accuracy5=model1.score(x_test,y_test)

print(accuracy5*100,'%')
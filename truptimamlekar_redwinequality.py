import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score

from pandas.plotting import scatter_matrix
df1=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df1.head()
df1.describe()
df1.info()
df1.isnull().values.any()
plt.bar(df1['quality'],df1['alcohol'],color='pink')

plt.title('Barh plot')

plt.xlabel ('quality')

plt.ylabel ('alcohol')

plt.legend()

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



df1.hist(bins=10, figsize=(20,20))

plt.show()
corr_matrix=df1.corr()

figure,ax=plt.subplots(figsize=(10,10))

sns.heatmap(corr_matrix, annot=True,linewidth=0.5,ax=ax)
f,ax=plt.subplots(1,1,figsize=(25,6))

ax = sns.scatterplot(x="volatile acidity", y="quality",color = "red",data=df1)

ax = sns.scatterplot(x="citric acid", y="quality",color = "green",data=df1)

ax = sns.scatterplot(x="chlorides", y="quality",color = "blue",data=df1)
f,ax=plt.subplots(1,1,figsize=(25,6))

sns.kdeplot(df1.loc[(df1['quality']==4), 'alcohol'], color='b', shade=True, Label='4')

sns.kdeplot(df1.loc[(df1['quality']==5), 'alcohol'], color='g', shade=True, Label='5')

sns.kdeplot(df1.loc[(df1['quality']==6), 'alcohol'], color='r', shade=True, Label='6')

sns.kdeplot(df1.loc[(df1['quality']==7), 'alcohol'], color='y', shade=True, Label='7')

plt.xlabel('Alcohol') 

plt.ylabel('Probability Density') 
sns.pairplot (df1)
f,ax=plt.subplots(1,3,figsize=(25,6))

box1=sns.boxplot(data=df1['fixed acidity'],ax=ax[0], palette="muted",sym='k.')

ax[0].set_xlabel('fixed acidity')

box1=sns.boxplot(data=df1['volatile acidity'],ax=ax[1], palette="muted",sym='k.')

ax[1].set_xlabel('volatile acidity')

box1=sns.boxplot(data=df1['citric acid'],ax=ax[2], palette="muted",sym='k.')

ax[2].set_xlabel('citric acid')

plt.show ()
f,ax=plt.subplots(1,3,figsize=(25,4))

box1=sns.boxplot(data=df1['residual sugar'],ax=ax[0], palette="muted",sym='k.')

ax[0].set_xlabel('residual sugar')

box1=sns.boxplot(data=df1['chlorides'],ax=ax[1], palette="muted",sym='k.')

ax[1].set_xlabel('chlorides')

box1=sns.boxplot(data=df1['free sulfur dioxide'],ax=ax[2], palette="muted",sym='k.')

ax[2].set_xlabel('free sulfur dioxide')

plt.show ()
f,ax=plt.subplots(1,3,figsize=(25,4))

box1=sns.boxplot(data=df1['total sulfur dioxide'],ax=ax[0], palette="muted",sym='k.')

ax[0].set_xlabel('total sulfur dioxide')

box1=sns.boxplot(data=df1['density'],ax=ax[1], palette="muted",sym='k.')

ax[1].set_xlabel('density')

box1=sns.boxplot(data=df1['pH'],ax=ax[2], palette="muted",sym='k.')

ax[2].set_xlabel('pH')

plt.show ()
f,ax=plt.subplots(1,3,figsize=(25,4))

box1=sns.boxplot(data=df1['sulphates'],ax=ax[0], palette="muted",sym='k.')

ax[0].set_xlabel('sulphates')

box1=sns.boxplot(data=df1['alcohol'],ax=ax[1], palette="muted",sym='k.')

ax[1].set_xlabel('alcohol')

box1=sns.boxplot(data=df1['quality'],ax=ax[2], palette="muted",sym='k.')

ax[2].set_xlabel('quality')

plt.show ()
f,ax=plt.subplots(2,2,figsize=(25,15))

sns.violinplot(x="quality", y="fixed acidity",ax=ax[0][0],data=df1, palette="muted")

sns.violinplot(x="quality", y="volatile acidity",data=df1,ax=ax[0][1], palette="muted")

sns.violinplot(x="quality", y="citric acid",ax=ax[1][0],data=df1, palette="muted")

sns.violinplot(x="quality", y="residual sugar",ax=ax[1][1],data=df1, palette="muted")
quality = df1["quality"].values

category = []

for num in quality:

    if num < 5.5:

        category.append("Low")

    else:

        category.append("High")
[(i, category.count(i)) for i in set(category)]

plt.figure(figsize=(10, 6))

sns.countplot(category, palette="muted")
features=df1.columns

features=features.drop(['quality'])

x=df1 [features]

y=df1 ['quality']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)
seed=7

models = []

models.append(('RF',RandomForestClassifier()))

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
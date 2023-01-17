#Importing required libraries.

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import metrics

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

#Loading dataset

wine = pd.read_csv("../input/wine-quality/winequality-red.csv")

wine
#Let's check top five obseravtion

wine.head()
#lets check the data types of columns and some info about data

wine.info()
#quality vs fixed acidity

plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)
# quality vs volatile acidity

plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)
#quality vs citric acid, if quality increase,citric acid will also increase

# highly positive corelated

plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'citric acid', data = wine)
plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'residual sugar', data = wine)
#quality vs chlorides,if quality increase,chlorides will decrease

# negative corelated

plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'chlorides', data = wine)
#nothing much is known

plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)
plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)
#if quality of wine increase,sulphates will also increase

plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'sulphates', data = wine)
#if quality of wine increase,alcohol level will also increase

plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'alcohol', data = wine)
# check null values

wine.isnull().sum()
wine.head()
wine.quality.value_counts()
#Create an empty list called Levels

Levels=[]

for i in wine['quality']:

    if i >= 1 and i <= 5:

        Levels.append('1')

    elif i >= 6 and i <= 10:

        Levels.append('2')

wine['Levels'] = Levels
#view final data

wine.info()
print(wine.Levels.unique())

print(wine.Levels.value_counts())
wine.shape
# i drop quality feature because inplace of quality i create levels feature

wine.drop("quality",axis=1,inplace=True)
#Now seperate the dataset into x and y variable

x = wine.drop('Levels', axis = 1)

y = wine['Levels']
#split the data into train and test  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
x_train.shape
x_test.shape
#Apply standard scaling to get optimized result

sc = StandardScaler()
x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=0)

rfc.fit(x_train,y_train)

y_pred_rf = rfc.predict(x_test)
#let's model accuracy

metrics.accuracy_score(y_test,y_pred_rf)
#Let's see model performance

print(metrics.classification_report(y_test,y_pred_rf))
from sklearn.svm import SVC

svc = SVC(random_state=0)

svc.fit(x_train,y_train)

y_pred_svc = svc.predict(x_test)
# let's check model accuracy

metrics.accuracy_score(y_test,y_pred_svc)
#let's check model classification report 

print(metrics.classification_report(y_test,y_pred_svc))
#Best parameters for our SVC model

param = {

    "C": [0.1,0.2,0.5,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5],

    "kernel":["linear","rbf"],

    "gamma" :[0.1,0.2,0.5,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]

}

gridCV_svc = GridSearchCV(svc, param_grid=param, scoring="accuracy", cv=10)
gridCV_svc.fit(x_train,y_train)
#Best parameters

gridCV_svc.best_params_
#Let's run SVC again with best parameters.

svc_best = SVC(C=0.9,gamma=0.5,kernel="rbf")

svc_best.fit(x_train,y_train)

y_pred_svc_best = svc_best.predict(x_test)

print(metrics.classification_report(y_test,y_pred_svc_best))
#Now let's try cross validation score for RF classifier.

rfc_CV = cross_val_score(estimator=rfc,X=x_train,y=y_train,cv=10)

rfc_CV.mean().round(2)
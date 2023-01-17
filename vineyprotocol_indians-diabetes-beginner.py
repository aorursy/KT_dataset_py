# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualiztion
%matplotlib inline
import seaborn as sns #sub libraray of matplotlib for visualizatoion


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read data
data=pd.read_csv("../input/diabetes/diabetes.csv")
data.head()
data.Outcome.value_counts().plot(kind="bar")
print(data.Outcome.value_counts())
# let's see the distribution of all variable except outcom
fig, ax = plt.subplots(4,2, figsize=(20,20))
sns.distplot(data.Age, bins = 20, ax=ax[0,0]) 
sns.distplot(data.Pregnancies, bins = 20, ax=ax[0,1]) 
sns.distplot(data.Glucose, bins = 20, ax=ax[1,0]) 
sns.distplot(data.BloodPressure, bins = 20, ax=ax[1,1]) 
sns.distplot(data.SkinThickness, bins = 20, ax=ax[2,0])
sns.distplot(data.Insulin, bins = 20, ax=ax[2,1])
sns.distplot(data.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 
sns.distplot(data.BMI, bins = 20, ax=ax[3,1]) 
sns.pairplot(data,hue="Outcome")
cor=data.corr()
plt.figure(figsize=(10,6))
sns.heatmap(cor,annot=True,cmap="RdYlGn")
data.head()
x=data.iloc[:,:-1]
x.head()
y=data.Outcome
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
print("75% train data" ,x_train.shape)
print("25% test data" ,x_test.shape)
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore") 
from sklearn import metrics
#Model
L_reg = LogisticRegression()

#fit the model
L_reg.fit(x_train, y_train)

#prediction
y_pred = L_reg.predict(x_test)

#Accuracy
print("Accuracy ", metrics.accuracy_score(y_test,y_pred))


# Plot confusion matrix
cm=metrics.confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="g")
from sklearn.tree import DecisionTreeClassifier

D_tree = DecisionTreeClassifier(random_state=0)

#fit the model
D_tree.fit(x_train, y_train)

#prediction
y_pred_dt = D_tree.predict(x_test)

#Accuracy
print("Accuracy ", metrics.accuracy_score(y_test,y_pred_dt))

#plot confusion matrix
cm = metrics.confusion_matrix(y_test,y_pred_dt)
sns.heatmap(cm, annot=True, fmt='g')
plt.show()
from sklearn.ensemble import RandomForestClassifier

Rf_model = RandomForestClassifier(random_state=0)

#fit the model
Rf_model.fit(x_train, y_train)

#prediction
y_pred_rf = Rf_model.predict(x_test)

#Accuracy
print("Accuracy ", metrics.accuracy_score(y_test,y_pred_rf))

#plot confusion matrix
cm = metrics.confusion_matrix(y_test,y_pred_rf)
sns.heatmap(cm, annot=True, fmt='g')
plt.show()
# check all model accuracy

print("Accuracy Logistic Reg ", metrics.accuracy_score(y_test,y_pred))
print("Accuracy Decission tree ", metrics.accuracy_score(y_test,y_pred_dt))
print("Accuracy Random Forest ", metrics.accuracy_score(y_test,y_pred_rf))

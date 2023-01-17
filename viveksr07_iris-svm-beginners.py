#Lets import important libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#Now lets read the data

df=pd.read_csv(r'../input/iris/Iris.csv')

df.drop('Id',inplace=True,axis=1)  #drop Id column 

df.head()

#lets explore data

df.info()

df.describe()

#Now lets create a plot to know more about data

sns.set_style('darkgrid')

sns.pairplot(df,hue='Species',palette='Dark2')
#Train Test Split

#now lets split train and test

from sklearn.model_selection import train_test_split



X=df.drop('Species',axis=1)

y=df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
#Now lets train the model and predict

#Now its time to train a Support Vector Machine Classifier.

from sklearn.svm import SVC

svc=SVC()

svc.fit(X_train,y_train)

#Now get predictions and model evaluation

pred=svc.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print(accuracy_score(y_test,pred))

print('\n')

print(classification_report(y_test,pred))

print('\n')

print(confusion_matrix(y_test,pred))
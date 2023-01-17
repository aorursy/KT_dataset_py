#titanic

#importing libriaries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#importing training dataset

dataset=pd.read_csv('train.csv')

x=dataset.iloc[:,[2,4,5,6,7]].values

y=dataset.iloc[:,1].values



#msissing data

from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

imputer=imputer.fit(x[:,2:3])

x[:,2:3]=imputer.transform(x[:,2:3])



#encoding categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_x=LabelEncoder()

x[:,1]=labelencoder_x.fit_transform(x[:,1])

onehotencoder=OneHotEncoder(categorical_features=[1])

x=onehotencoder.fit_transform(x).toarray()

x=x[:,1:]



#feature scaling

from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()

x=sc_x.fit_transform(x)



#fitting logistic regression to training set

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0)

classifier.fit(x,y)



#importing testing dataset

dataset2=pd.read_csv('test.csv')

x_test=dataset2.iloc[:,[1,3,4,5,6]].values

datase=pd.read_csv('gender_submission.csv')

y_test=datase.iloc[:,1].values



#msissing data

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

imputer=imputer.fit(x_test[:,2:3])

x_test[:,2:3]=imputer.transform(x_test[:,2:3])



#encoding categorical data

labelencoder_x=LabelEncoder()

x_test[:,1]=labelencoder_x.fit_transform(x_test[:,1])

onehotencoder=OneHotEncoder(categorical_features=[1])

x_test=onehotencoder.fit_transform(x_test).toarray()

x_test=x_test[:,1:]



#feature scaling

x_test=sc_x.transform(x_test)



#predicting the test set results

y_pred=classifier.predict(x_test)



#making the confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
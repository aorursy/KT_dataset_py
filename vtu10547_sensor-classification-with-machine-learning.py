import numpy as np

import pandas as pd 

from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

data=pd.read_csv("../input/iot-sensorscsv/Sensors.csv")
df=pd.DataFrame(data)

df
Y=df['Output']



Y
lb=preprocessing.LabelEncoder()



lb.fit(Y)

Y=lb.transform(Y)
Y
X=df.drop('Output',axis=1)
X
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)
model=MLPClassifier()

model.fit(X,Y)
LR=LogisticRegression()

LR.fit(X,Y)
rdf=RandomForestClassifier()

rdf.fit(X_train,Y_train)
from sklearn.metrics import accuracy_score

print(accuracy_score(Y,model.predict(X)))

print(accuracy_score(Y,LR.predict(X)))

print(accuracy_score(Y_test,rdf.predict(X_test)))
X
if((rdf.predict([[52,3]]))==2):

  print("ULtrasonic Sensor")

if((rdf.predict([[25,2]]))==0):

  print("DHT Sensor")

if((rdf.predict([[5,0.01]]))==1):

  print("PH Sensor")
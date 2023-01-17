import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf

%matplotlib inline
X_train=pd.read_csv('../input/airline-passenger-satisfaction/train.csv')
X_train.head()
X_test=pd.read_csv('../input/airline-passenger-satisfaction/test.csv')
X_test.head(10)
X_train.columns
corr=X_train.corr()

corr_d=corr['Departure Delay in Minutes'].sort_values(ascending=False)
plt.figure(figsize=(11,10))

sns.heatmap(corr,linewidths=0.6,annot=True,fmt='.1f',cmap='coolwarm',square=True,robust=True)
#checking relation between departure and arrival delay.

plt.figure(figsize=(9,7))

sns.scatterplot('Departure Delay in Minutes','Arrival Delay in Minutes',data=X_train,color='g')
X_train.dropna(axis=0,subset=['satisfaction'],inplace=True)

X_test.dropna(axis=0,subset=['satisfaction'],inplace=True)
plt.figure(figsize=(9,6))

plt.subplot(1,2,1)

sns.countplot('Inflight service',data=X_train,hue='Customer Type')

plt.subplot(1,2,2)

sns.countplot('Cleanliness',data=X_train,hue='Customer Type')

#here we check variation in ratings,prominently people of Business class have given good feedback compared to eco and Eco pls calsses.

plt.figure(figsize=(9,6))

plt.subplot(1,2,1)

sns.countplot('On-board service',data=X_train,hue='Class')

plt.subplot(1,2,2)

sns.countplot('Leg room service',data=X_train,hue='Class')
#from here we can infer that mostly Business class people have been priortized followed by Eco class with

#almost same importance.

plt.figure(figsize=(9,6))

plt.subplot(1,2,1)

sns.countplot('Inflight entertainment',data=X_train,hue='Class')

plt.subplot(1,2,2)

sns.countplot('Checkin service',data=X_train,hue='Class')
y_train=X_train.satisfaction

X_train=X_train.drop('satisfaction',axis=1)

y_test=X_test.satisfaction

X_test=X_test.drop('satisfaction',axis=1)
#preprocess the data 

from category_encoders import CountEncoder

ce=CountEncoder()

ce_train=pd.DataFrame(ce.fit_transform(X_train,y_train))

ce_test=pd.DataFrame(ce.transform(X_test,y_test))
ce_train=ce_train.fillna(0)

ce_test=ce_test.fillna(0)
from catboost import CatBoostClassifier

cat=CatBoostClassifier(learning_rate=0.09)

cat.fit(ce_train,y_train)

preds=cat.predict(ce_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,preds))
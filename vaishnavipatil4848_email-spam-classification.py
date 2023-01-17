import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf

%matplotlib inline
X=pd.read_csv('../input/email-spam-classification-dataset-csv/emails.csv')
X.head()
plt.figure(figsize=(9,7))

sns.lineplot('Prediction','a',hue='ect',data=X,palette='viridis')
X.dropna(axis=0,subset=['Prediction'],inplace=True)

y=X.Prediction

X=X.drop('Prediction',axis=1)
X.head()
y.head()
plt.figure(figsize=(9,8))

sns.lineplot('you','the',data=X,marker='o',markerfacecolor='blue',markersize='10')
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)
from sklearn.preprocessing import LabelEncoder

encode=LabelEncoder()



object_cols=[col for col in X_train.columns if

            X_train[col].dtype=="object"]

good_cols=[col for col in object_cols if

          set(X_train[col])==set(X_test[col])]

bad_cols=list(set(object_cols)-set(good_cols))
label_train=X_train.drop(bad_cols,axis=1)

label_test=X_test.drop(bad_cols,axis=1)
label_train.head()
for col in good_cols:

    label_train[col]=pd.DataFrame(encode.fit_transform(label_train[col]))

    label_test[col]=pd.DataFrame(encode.transform(label_test[col]))
label_train.head()
#scaling the training samples to avoid samples which have extreme values

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaled_f=scaler.fit_transform(label_train)

scaled_f1=scaler.fit_transform(label_test)
label_train_sc=pd.DataFrame(scaled_f,columns=label_train.columns)

label_test_sc1=pd.DataFrame(scaled_f1,columns=label_test.columns)
#using randomforest for making predictions

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=150)

model.fit(label_train_sc,y_train)

preds=model.predict(label_test_sc1)
print(classification_report(y_test,preds))

print(confusion_matrix(y_test,preds))
label_train_sc.shape
from catboost import CatBoostClassifier

cb=CatBoostClassifier(learning_rate=0.1)

cb.fit(label_train_sc,y_train)

pred=cb.predict(label_test_sc1)
print(classification_report(y_test,pred))
#analysing predictions vs actual values

analyse=pd.DataFrame(y_test,pred)
plt.figure(figsize=(8,7))

sns.lineplot(y_test,pred)
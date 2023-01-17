import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
data=pd.read_csv("../input/creditcardfraud/creditcard.csv")
data.head()
X=data.drop("Class",axis=1)
Y=data["Class"]
X.head()
## Get the Fraud and the normal dataset 

fraud = data[data['Class']==1]

normal = data[data['Class']==0]
print(fraud.shape,normal.shape)
from imblearn.under_sampling import NearMiss
nm = NearMiss(sampling_strategy='auto')
X_res,Y_res=nm.fit_sample(X,Y)
X_res.shape,Y_res.shape
from sklearn.linear_model import LogisticRegression
LR_model=LogisticRegression()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_res,Y_res,random_state=1,test_size=0.2)
LR_model.fit(X_train,y_train)
predict=LR_model.predict(X_test)
from sklearn.metrics import accuracy_score,recall_score
accuracy=accuracy_score(y_test,predict)
print(accuracy)
print(recall_score(y_test,predict))
print(fraud.shape,normal.shape)
from imblearn.combine import SMOTETomek
smk=SMOTETomek(random_state=42)
X_up,Y_up=smk.fit_sample(X,Y)
X_up.shape,Y_up.shape
Y_up.value_counts()
X1_train,X1_test,Y1_train,Y1_test=train_test_split(X_up,Y_up,random_state=42,test_size=0.2)
X1_train.shape
LR_model
LR_model.fit(X1_train,Y1_train)
predict1=LR_model.predict(X1_test)
print(accuracy_score(predict1,Y1_test))
print(recall_score(predict1,Y1_test))
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
X2_train=X1_train.sample(frac=0.3,random_state=42)
X2_test=X1_test.sample(frac=0.3,random_state=42)
Y2_train=Y1_train.sample(frac=0.3,random_state=42)
Y2_test=Y1_test.sample(frac=0.3,random_state=42)
X2_train.shape
rf.fit(X2_train,Y2_train)
predict2=rf.predict(X2_test)
print(accuracy_score(Y2_test,predict2))
print(recall_score(Y2_test,predict2))

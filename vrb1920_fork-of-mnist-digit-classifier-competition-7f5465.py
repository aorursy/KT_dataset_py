# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head()

##Spliting the train set into train1 and validate
from sklearn.model_selection import train_test_split
train1,validate=train_test_split(train,test_size=.3,random_state=100)
##prepration of data 
train1_x=train1.drop("label",axis=1)
train1_y=train1['label']
test1_x=validate.drop("label",axis=1)
test1_y=validate["label"]
print(train1_x.shape)
print(train1_y.shape)
print(test1_x.shape)
print(test1_y.shape)

##Decission tree classifier
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=100)
model.fit(train1_x,train1_y)

##predicting the test data
test_pred=model.predict(test1_x)
df_pred=pd.DataFrame({"actual":test1_y,"predicted":test_pred})
df_pred['pred_status']=df_pred['predicted']==df_pred['actual']
accuracy=df_pred[df_pred['pred_status']==True].shape[0]/df_pred.shape[0]*100
accuracy
#accu=(tp+tn)/(tn+tp+fp+fn)
#print(accu)

##random forest
from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier(random_state=100,n_estimators=1000)
model1.fit(train1_x,train1_y)
test_pred1=model1.predict(test1_x)
df_pred1=pd.DataFrame({'actual':test1_y,"predicted":test_pred1})
df_pred1['pred_status']=df_pred1['actual']==df_pred1["predicted"]
accuracy1=df_pred1[df_pred1['pred_status']==True].shape[0]/df_pred1.shape[0]*100
accuracy1

##ada boost
from sklearn.ensemble import AdaBoostClassifier
model2=AdaBoostClassifier(random_state=100)
model2.fit(train1_x,train1_y)
test_pred2=model2.predict(test1_x)
df_pred2=pd.DataFrame({'actual':test1_y,'predicted':test_pred2})
df_pred2['pred_status']=df_pred2['actual']==df_pred2['predicted']
accuracy2=df_pred2[df_pred2['pred_status']==True].shape[0]/df_pred2.shape[0]*100
accuracy2

##KNeighbours
from sklearn.neighbors import KNeighborsClassifier
model3=KNeighborsClassifier(n_neighbors=5)
model3.fit(train1_x,train1_y)
test_pred3=model3.predict(test1_x)
df_pred3=pd.DataFrame({'actual':test1_y,'predicted':test_pred3})
df_pred3['pred_status']=df_pred2['actual']==df_pred3['predicted']
accuracy3=df_pred3[df_pred3['pred_status']==True].shape[0]/df_pred3.shape[0]*100
accuracy3

### working on original datasets
## preparation of data 
train_x=train.drop("label",axis=1)
train_y=train["label"]
m=RandomForestClassifier(random_state=100,n_estimators=1000)
m.fit(train_x,train_y)
test_predict=m.predict(test)
dt_predict=pd.DataFrame({"Label":test_predict})
dt_predict["ImageId"]=test.index+1
dt_predict[["ImageId","Label"]].to_csv("Submission.csv",index=False)



import numpy as np 

import pandas as pd





import os

print(os.listdir("../input"))



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score 
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
X=train.iloc[ : ,2:202].values
y=train.iloc[ :,1:2].values

y=y.astype(float)
train.head()
train[train.columns[1:]].corr()['target'][:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
reg= LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
y_pred=y_pred.round()

accuracy_score(y_pred,y_test)
y1=train.iloc[ :,0:202]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size = 0.2, random_state = 0)
y_test1=y_test1.reset_index()

y_pred=pd.DataFrame(y_pred)
res=pd.concat([y_pred,y_test1],axis=1)
res=res.rename(columns={0:"predicted"})
res['diff']=res['predicted']-res['target']
res["diff"]=res["diff"].abs()
wr_pred=res[res["diff"]==1]
wr_pred.head()
X_TRAIN=train.iloc[ : ,2:202].values
Y_TRAIN=train.iloc[ :,1:2].values

Y_TRAIN=Y_TRAIN.astype(float)
X_TEST=test.iloc[ :,1:201].values
reg.fit(X_TRAIN,Y_TRAIN)
TARGET=reg.predict(X_TEST)
TARGET=TARGET.round()
TARGET=pd.DataFrame(TARGET)
TARGET.shape
ID_CODE=test.iloc[ :,0:1]
RESULT=pd.concat([ID_CODE,TARGET],axis=1)
RESULT=RESULT.rename(columns={0:'target'})
RESULT["target"]=RESULT["target"].abs()
RESULT.head()
RESULT.to_csv("test_target_pred.csv")
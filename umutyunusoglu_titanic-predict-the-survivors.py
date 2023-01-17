import numpy as np # for arrays and algebra

import pandas as pd # for data processing



#For visualising data

import matplotlib

import matplotlib.pyplot as plt



#For Data Preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import  train_test_split









import os #for operating system operations

print(os.listdir("../input")) #Here we list the files in input folder

dataset=pd.read_csv("../input/train.csv")#We read our data

test_set=pd.read_csv("../input/test.csv")





dataset.head()
test_set.head()
dataset=dataset.filter(["Survived","Pclass","Sex","Age","SibSp","Parch","Cabin","Embarked"])



test_id=test_set.iloc[:,0].values

test_set=test_set.filter(["Survived","Pclass","Sex","Age","SibSp","Parch","Cabin","Embarked"])

test_id
le=LabelEncoder()

dataset.iloc[:,2]=le.fit_transform(dataset.iloc[:,2])



dataset["Age"].fillna(dataset["Age"].mean(),inplace=True)

dataset['Cabin'] = dataset['Cabin'].apply(lambda x: 1 if not pd.isnull(x) else 0)



def Embark(x):

    if x=="C":

        return 0

    elif x=="S":

        return 1

    elif x=="Q":

        return 2

    else:

        return 3



dataset["Embarked"]=dataset["Embarked"].apply(Embark)

dataset.head()

le=LabelEncoder()

test_set.iloc[:,1]=le.fit_transform(test_set.iloc[:,1])



test_set["Age"].fillna(test_set["Age"].mean(),inplace=True)

test_set['Cabin'] = test_set['Cabin'].apply(lambda x: 1 if not pd.isnull(x) else 0)



def Embark(x):

    if x=="C":

        return 0

    elif x=="S":

        return 1

    elif x=="Q":

        return 2

    else:

        return 3



test_set["Embarked"]=test_set["Embarked"].apply(Embark)



test_set.head()
X=dataset.iloc[:,1:]

Y=dataset.iloc[:,0]



x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33)
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



knn=KNeighborsRegressor(n_neighbors=3)

knn.fit(x_train,y_train)

knn_pred=np.round(knn.predict(x_test))

knn_cm=confusion_matrix(y_test,knn_pred)

knn_acc=(knn_cm[0][0]+knn_cm[1][1])/(knn_cm[0][0]+knn_cm[1][1]+knn_cm[0][1]+knn_cm[1][0])



lr=LinearRegression()

lr.fit(x_train,y_train)

lr_pred=np.round(lr.predict(x_test))

lr_cm=confusion_matrix(y_test,lr_pred)

lr_acc=(lr_cm[0][0]+lr_cm[1][1])/(lr_cm[0][0]+lr_cm[1][1]+lr_cm[0][1]+lr_cm[1][0])



lor=LogisticRegression()

lor.fit(x_train,y_train)

lor_pred=np.round(lor.predict(x_test))

lor_cm=confusion_matrix(y_test,lor_pred)

lor_acc=(lor_cm[0][0]+lor_cm[1][1])/(lor_cm[0][0]+lor_cm[1][1]+lor_cm[0][1]+lor_cm[1][0])



svr=SVR(kernel="rbf")

svr.fit(x_train,y_train)

svr_pred=np.round(svr.predict(x_test))

svr_cm=confusion_matrix(y_test,svr_pred)

svr_acc=(svr_cm[0][0]+svr_cm[1][1])/(svr_cm[0][0]+svr_cm[1][1]+svr_cm[0][1]+svr_cm[1][0])



rf=RandomForestRegressor(n_estimators=10)

rf.fit(x_train,y_train)

rf_pred=np.round(rf.predict(x_test))

rf_cm=confusion_matrix(y_test,rf_pred)

rf_acc=(rf_cm[0][0]+rf_cm[1][1])/(rf_cm[0][0]+rf_cm[1][1]+rf_cm[0][1]+rf_cm[1][0])



xgb=XGBRegressor()

xgb.fit(x_train,y_train)

xgb_pred=np.round(xgb.predict(x_test))

xgb_cm=confusion_matrix(y_test,xgb_pred)

xgb_acc=(xgb_cm[0][0]+xgb_cm[1][1])/(xgb_cm[0][0]+xgb_cm[1][1]+xgb_cm[0][1]+xgb_cm[1][0])



print(knn_acc)

print(lr_acc)

print(lor_acc)

print(svr_acc)

print(rf_acc)

print(xgb_acc)
classifier=XGBRegressor()

classifier.fit(X,Y)



result=np.round(classifier.predict(test_set))

result



data=list(zip(test_id,result))

submission=pd.DataFrame(data=data,columns=["PassengerId","Survived"])

submission.to_csv("result.csv",index=False)

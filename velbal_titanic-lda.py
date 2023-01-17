import pandas as pd
import numpy  as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn                       import metrics
# Define parameters
FCSV_TRAIN="../input/train.csv"
FCSV_TESTX="../input/test.csv"
FCSV_TESTY="../input/gender_submission.csv"
Y="Survived"
REMOVE=["PassengerId","Name","Sex","Embarked","Ticket","Cabin","Age"]
# Load training data from train.csv
data=pd.read_csv(FCSV_TRAIN)
data=pd.concat([data.drop(REMOVE,axis=1),pd.get_dummies(data['Sex']),      \
                                         pd.get_dummies(data['Embarked'])],axis=1)
data=data.drop(['female'],axis=1)
data=data.drop(['C']     ,axis=1)
data=data.dropna()

x_train=data.drop([Y],axis=1)
y_train=data[Y]
x_train_ave=x_train.mean(axis=0)
y_train_ave=y_train.mean(axis=0)
x_train_std=x_train.std(axis=0,ddof=1)
y_train_std=y_train.std(axis=0,ddof=1)
# Auto scaling training data with mean=0 and var=1 
x_train['Pclass']=(x_train['Pclass']-x_train['Pclass'].mean(axis=0))/x_train['Pclass'].std(axis=0,ddof=1)
x_train['Parch'] =(x_train['Parch'] -x_train['Parch'].mean(axis=0)) /x_train['Parch'].std(axis=0 ,ddof=1)
x_train['SibSp'] =(x_train['SibSp'] -x_train['SibSp'].mean(axis=0)) /x_train['SibSp'].std(axis=0 ,ddof=1)
x_train['Fare']  =(x_train['Fare']  -x_train['Fare'].mean(axis=0))  /x_train['Fare'].std(axis=0  ,ddof=1)
# Load test data from test.csv and gender_submission.csv
data_testx=pd.read_csv(FCSV_TESTX)
data_testy=pd.read_csv(FCSV_TESTY)
data_test=pd.concat([data_testy,data_testx],axis=1)
data_test=pd.concat([data_test.drop(REMOVE,axis=1),pd.get_dummies(data_test['Sex']),      \
                                                   pd.get_dummies(data_test['Embarked'])],axis=1)
data_test=data_test.drop(['female'],axis=1)
data_test=data_test.drop(['C'],axis=1)
data_test=data_test.dropna()

x_test=data_test.drop(['Survived'],axis=1)
y_test=data_test['Survived']
# Auto scaling test data with mean=0 and var=1 
x_test['Pclass']=(x_test['Pclass']-x_train_ave['Pclass'])/x_train_std['Pclass']
x_test['Parch'] =(x_test['Parch'] -x_train_ave['Parch']) /x_train_std['Parch']
x_test['SibSp'] =(x_test['SibSp'] -x_train_ave['SibSp']) /x_train_std['SibSp']
x_test['Fare']  =(x_test['Fare']  -x_train_ave['Fare'])  /x_train_std['Fare']
model=LDA()
model.fit(x_train,y_train)
# Predicting training data with the model constructed 
yp_train=model.predict(x_train)
accuracy=metrics.accuracy_score(y_train,yp_train)
confusin=metrics.confusion_matrix(y_train,yp_train)
print(accuracy)
print(confusin)
# Predicting test data with the model constructed 
yp_test =model.predict(x_test)
accuracy=metrics.accuracy_score(y_test,yp_test)
confusin=metrics.confusion_matrix(y_test,yp_test)
print(accuracy)
print(confusin)
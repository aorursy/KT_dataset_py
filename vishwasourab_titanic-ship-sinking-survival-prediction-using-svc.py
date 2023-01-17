#import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
#import test and train data
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
#define a function that loads train file, takes important columns, label encodes sex and fill missing values in age column
#this function returns survived(y) and data(x) used to train the model 
def load_titanic_train_file():
    titanic_train_file=pd.read_csv('../input/train.csv')
    cols=["Pclass","Sex","Age"]
    #change male to 1 and female to 0
    titanic_train_file["Sex"]=titanic_train_file["Sex"].apply(lambda sex:1 if sex=="male" else 0)
    #handle missing values of age
    titanic_train_file["Age"]=titanic_train_file["Age"].fillna(titanic_train_file["Age"].mean())
    titanic_train_file["Age"]=titanic_train_file["Fare"].fillna(titanic_train_file["Fare"].mean())
    survived=titanic_train_file["Survived"].values
    data=titanic_train_file[cols].values
    return survived, data
#define a function that loads test file, takes important columns, label encodes sex and fill missing values in age column
#this function returns only data(x) used to predict the survival 
def load_titanic_test_file():
    titanic_train_file=pd.read_csv('../input/test.csv')
    cols=["Pclass","Sex","Age"]
    #change male to 1 and female to 0
    titanic_train_file["Sex"]=titanic_train_file["Sex"].apply(lambda sex:1 if sex=="male" else 0)
    #handle missing values of age
    titanic_train_file["Age"]=titanic_train_file["Age"].fillna(titanic_train_file["Age"].mean())
    titanic_train_file["Age"]=titanic_train_file["Fare"].fillna(titanic_train_file["Fare"].mean())
    data=titanic_train_file[cols].values
    return data
# Create a ML Model
survived, data_train=load_titanic_train_file()
data_test=load_titanic_test_file()
# SVC - linear Kernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model=SVC(kernel='linear',C=1.0)
model.fit(data_train,survived)
predictValuesTrain=model.predict(data_train)
acc=model.score(data_train,survived)
print(acc)

# SVC - rbf Kernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model1=SVC(kernel='rbf',C=1.5)
model1.fit(data_train,survived)
#predictValuesTrain=model.predict(data_train)
acc1=model1.score(data_train,survived)
print(acc1)

#predict test values and output to a file
#predict values for test data set and output them to a file
predictValuesTest=model1.predict(data_test)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predictValuesTest
    })
submission.to_csv('submission.csv', index=False)

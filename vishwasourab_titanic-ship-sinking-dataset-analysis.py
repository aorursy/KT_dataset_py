#import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# load train and test files
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
#view first 5 records of train file
train_df.head()
#view first 5 records of test file
test_df.head()
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
#create dataset for ML
survived,data_train=load_titanic_train_file()
data_test=load_titanic_test_file()
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(data_train,survived)
predictValues=model.predict(data_test)
predictValues
#output to a file
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predictValues
    })
submission.to_csv('submission.csv', index=False)

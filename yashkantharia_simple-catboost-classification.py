#Importing the libraries

import numpy as np

import pandas as pd

from catboost import CatBoostClassifier



# Importing the dataset

dataset = pd.read_csv('../input/train.csv', encoding= 'latin-1')

test = pd.read_csv('../input/test.csv', encoding= 'latin-1')



#removing na values

dataset.fillna(0, inplace=True)

test.fillna(0, inplace=True)



#Training data 

cols=[0,2,4,5,6,7,8,9,10,11]

X = dataset.iloc[:, cols].values

Y = dataset.iloc[:, 1].values



#Test data 

col=[0,1,3,4,5,6,7,8,9,10]

X_test = test.iloc[:,col].values



#Prediction

cat_feat = [2,6,8,9]

model=CatBoostClassifier()

model.fit(X, Y,cat_features=cat_feat)

y_pred = model.predict(X_test)



#save output in csv format

pd.DataFrame({'PassengerId': X_test[:,0],'Survived':y_pred}).to_csv("output.csv",index=False)
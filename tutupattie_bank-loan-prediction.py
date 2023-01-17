#import libraries
import numpy as np
import pandas as pd


#read data
customer_data = pd.read_csv('../input/bank-loan-classification/UniversalBank.csv')
customer_data.head()

#dropping unnecessary columns
customer_data.drop(['ID','ZIP Code'],axis = 1, inplace = True)
customer_data.head()

#splitting data 
from sklearn.model_selection import train_test_split
X = customer_data.copy().drop('Personal Loan',axis = 1)
Y = customer_data["Personal Loan"]
trainX,testX,trainY,testY = train_test_split(X,Y,test_size = 0.2)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

#training the model
from sklearn.svm import SVC

## Create an SVC object
svc = SVC()

## Fit
svc.fit(trainX,trainY)

## Predict
train_predictions = svc.predict(trainX)
test_predictions = svc.predict(testX)

# Train data accuracy
from sklearn.metrics import accuracy_score
print(f"Train data accuracy is {accuracy_score(trainY,train_predictions)}")
      
# Test data accuracy
print(f"Test data accuracy is {accuracy_score(testY,test_predictions)}")

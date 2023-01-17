import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import seaborn as sns
seed = 5
np.random.seed(seed)
from sklearn.datasets import load_boston
# Load the Boston Housing dataset from sklearn
boston = load_boston()
bos = pd.DataFrame(boston.data)
# give our dataframe the appropriate feature names
bos.columns = boston.feature_names
# Add the target variable to the dataframe
bos['Price'] = boston.target
# For student reference, the descriptions of the features in the Boston housing data set
# are listed below
boston.DESCR
bos.shape
bos.head()
bos.corr()
bos['CHAS'].tolist()
bos['RAD'].value_counts()

#bos=bos.drop(['AGE'],axis=1)
#bos=bos.drop(['B'],axis=1)
#check NA in every column
bos.isnull().sum(axis = 0)
# Select target (y) and features (X)
X = bos.iloc[:,:-1]
y = bos.iloc[:,-1]
print(X)
RAD_ = pd.get_dummies(X['RAD'], prefix='D_RAD', drop_first=True)
RAD_
X=pd.concat([X,RAD_],axis=1)
X
y
# Split the data into a train test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=seed, shuffle=True)
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(x_test)
#print(y_test,y_pred)
print(regressor.intercept_)
print(regressor.coef_)
#error
from math import sqrt
from sklearn.metrics import mean_squared_error
rms = sqrt(mean_squared_error(y_test, y_pred))
print(rms)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred,multioutput='raw_values')

#r2_score?
SS_Residual = sum((y_test-y_pred)**2)       
SS_Total = sum((y_test-np.mean(y_test))**2)     
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
print('r_squared:',r_squared)
print('adjusted_r_squared:',adjusted_r_squared)
#sns.heatmap(bos)

# Cross validation
from sklearn.model_selection import cross_val_score
regressor_kfold = LinearRegression()
scores = cross_val_score(regressor_kfold, X, y, cv=5)
print(scores.mean()*100.0)
y_pred = regressor.predict(x_test)
r2_score(y_test, y_pred,multioutput='raw_values')


bos

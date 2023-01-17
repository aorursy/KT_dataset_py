import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px



#importing libraries to use the library functions
train=pd.read_csv('../input/house-price-prediction-with-boston-housing-dataset/train1.csv')

#train contains the file information  which is in csv format
train
train.columns

#listing the column names of the dataset/dataframe
train.dtypes

#checking the datatypes of different columns of the dataframe
train.shape

#shape of the dataframe ie no. of rows and columns
train.duplicated().sum()

#checking for any duplicates in the data
train.isnull().sum()

#checking for any null values in the data
train.dropna(inplace=True,axis=0)

#removing the null values in the data
train.info()

#getting the information of dataframe such as no. of entries,data columns,non-null count,data types,etc
train.describe()

#checking for statistical summary such as count,mean,etc. of numeric columns
sns.boxplot(data=train,orient='h',palette='Set2')

#checking for any outliers in the data
sns.boxplot(x=train['AGE'])
sns.boxplot(x=train['TAX'])
sns.boxplot(x=train['Id'])
sns.boxplot(x=train['MEDV'])
Q1 = train.quantile(0.25)

Q3 = train.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
train.corr()

#finding the correlation between different variables/features
train_corr=train.corr()

f,ax=plt.subplots(figsize=(12,7))

sns.heatmap(train_corr,cmap='viridis',annot=True)

plt.title("Correlation between features",weight='bold',fontsize=18)

plt.show()



#plotting the heatmap for different features
X = train.drop('MEDV', axis = 1)

y = train['MEDV']
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



y = np.round(train['MEDV'])



#Apply SelectKBest class to extract top 5 best features

bestfeatures = SelectKBest(score_func=chi2, k=5)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)



# Concat two dataframes for better visualization

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score'] #naming the dataframe columns

featureScores
print(featureScores.nlargest(5,'Score')) #print 5 best features
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
# splitting the dataset into training and test sets



from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X_train,y_train)

y_predict = model.predict(X_train)



# calculating the accuracies

print("Training Accuracy :",model.score(X_train,y_train)*100)

print("Testing Accuracy :",model.score(X_test,y_test)*100)
from sklearn.metrics import mean_squared_error,r2_score

print("Model Accuracy",r2_score(y,model.predict(X))*100)
test=pd.read_csv('../input/house-price-prediction-with-boston-housing-dataset/test1.csv')

#test contains the file information  which is in csv format
test
test.shape
# Checking for missing values

test.isna().sum()
test['CRIM'] = test['CRIM'].fillna(test['CRIM'].mean())

test['ZN'] = test['ZN'].fillna(test['ZN'].mean())

test['INDUS'] = test['INDUS'].fillna(test['INDUS'].mean())

test['CHAS'] = test['CHAS'].fillna(test['CHAS'].mean())

test['AGE'] = test['AGE'].fillna(test['AGE'].mean())

test['LSTAT'] = test['LSTAT'].fillna(test['LSTAT'].mean())
y_pred = model.predict(test)
test['MEDV'] = y_pred
test
submission = test.drop(['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'],axis=1)

submission
submission.to_csv('submit.csv',index=False)
X = train.drop('MEDV', axis = 1)

y = train['MEDV']
# splitting the dataset into training and test sets



from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(X_train,y_train)

y_predict = rfr.predict(X_train)



# calculating the accuracies

print("Training Accuracy :",rfr.score(X_train,y_train)*100)

print("Testing Accuracy :",rfr.score(X_test,y_test)*100)
from sklearn.metrics import mean_squared_error,r2_score

print("Model Accuracy",r2_score(y,rfr.predict(X))*100)
test = test.drop('MEDV', axis = 1)

y_pred = rfr.predict(test)
test['MEDV'] = y_pred
test
submission = test.drop(['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'],axis=1)

submission
submission.to_csv('submit1.csv',index=False)
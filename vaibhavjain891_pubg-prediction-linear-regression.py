import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
test_df=pd.read_csv('../input/test.csv')

train_df=pd.read_csv('../input/train.csv')
train_df['winPlacePerc'].median()
# base line model
test_df['winPlacePerc'] = 0.45
df=pd.concat((train_df,test_df),axis=0)
df.describe()
train_df.walkDistance.plot(kind='box') 
X= df.iloc[:, 1:25].values
y= df.iloc[:,[25]].values
X,y
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
# # Create your regressor here
from sklearn import linear_model
regressor = linear_model.LinearRegression()
#Fitting the Regression Model to the dataset

regressor.fit(X_train,y_train)


y_pred = regressor.predict(X_test)
y_pred
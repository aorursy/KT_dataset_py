# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
sns.countplot(x = 'SeniorCitizen', data = df, hue = 'gender')
sns.countplot(x = 'Contract', data = df, hue = 'gender')
sns.countplot(x = 'DeviceProtection', data = df, hue = 'gender')
# Categorizing Male and Female in 1s and 0s
def gender_labels(element):
    if element == 'Male':
        return 0
    elif element == 'Female':
        return 1
# Making a new column in the dataframe
df['GenderLabel'] = df['gender'].apply(gender_labels)

#Dropping the original gender column
df.drop(['gender'] ,axis = 1, inplace=True)    
# Now, to relable the columns which have just "Yes" and "No" as their entries!
listOfColumns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'Churn', 'StreamingMovies', 'StreamingTV', 'DeviceProtection', 'PaperlessBilling']

# The Labelling Function
def Labelizer(input_value):
    '''Returns 1 for a Yes and a 0 for any other No'''
    if input_value == 'Yes':
        return 1
    else:
        return 0
    
for i in listOfColumns:
    newCol = i+'_label'
    df[newCol] = df[i].apply(Labelizer)

df.drop(listOfColumns, axis = 1, inplace=True)

list_nonBinary = ['Contract', 'PaymentMethod', 'InternetService']
for i in list_nonBinary:
    df = pd.concat([df, pd.get_dummies(df[i])], axis = 1)
    df.drop([i], axis = 1, inplace=True)

#print("Post feature Engineering, the columns are as follows : ", df.columns.values)
df['TotalChargesNew'] = df['tenure']*df['MonthlyCharges']
df.drop(['MonthlyCharges', 'TotalCharges'], axis = 1, inplace = True)
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
#Dropping the Customer ID for obvious reasons
df.drop(['customerID'], axis = 1, inplace=True)
from sklearn.model_selection import train_test_split
y = df['TotalChargesNew']
X = df.drop(['TotalChargesNew'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.tree import DecisionTreeRegressor
myRegressor = DecisionTreeRegressor(criterion='mse')
myRegressor.fit(X_train, y_train)
prediction = myRegressor.predict(X_test)
final_df_Decision = pd.DataFrame({'Predictions':prediction, 'True' : y_test})
final_df_Decision.head()
Prediction_Line = go.Scatter(
    x = [i for i in range(250)],
    y = prediction[:250]
)
Actual_Line = go.Scatter(
    x = [i for i in range(250)],
    y = y_test.values[:250]
)

data = [Prediction_Line, Actual_Line]
iplot(data)
from sklearn.metrics import mean_squared_error
print("Decision Tree metrics are about accurate to %.2f dollars (+ and -)"% (mean_squared_error(prediction, y_test)**0.5))
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.linear_model import LinearRegression
clf2 = LinearRegression()
clf2.fit(X_train, y_train)
preds2 = clf2.predict(X_test)

print("RMSE Score for Linear Regressor is %.2f"%(mean_squared_error(y_test, preds2))**0.5)

from sklearn.ensemble import GradientBoostingRegressor
clf3 = GradientBoostingRegressor()
clf3.fit(X_train, y_train)
preds3 = clf3.predict(X_test)

print("RMSE Score for Gradient Boost Regressor is %.2f"%(mean_squared_error(y_test, preds3))**0.5)
from xgboost import XGBRegressor
clf4 = XGBRegressor()
clf4.fit(X_train, y_train)
preds4 = clf4.predict(X_test)

print("RMSE Score of XGBoost Regressor is %.2f"%(mean_squared_error(y_test, preds4))**0.5)
from keras.models import Sequential
from keras.layers import (Dense, Dropout, BatchNormalization)

model = Sequential()
model.add(Dense(25, input_dim = 25, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.9))

model.add(Dense(12, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(6, activation='relu'))
model.add(Dropout(0.9))

model.add(Dense(1))
model.compile(optimizer='adam', loss= 'mse')

model.summary()
model.lr = 0.05
history = model.fit(X_train, y_train, epochs=1000, verbose=2)
benchmarks = pd.DataFrame({"Naive Deep Learning Model" : (sum(history.history['loss'])/len(history.history['loss']))**0.5,
                          "XG Boost Regressor" : mean_squared_error(y_test, preds4)**0.5,
                          "Gradiant Boost Regressor" : mean_squared_error(y_test, preds3)**0.5,
                          "Decision Tree Regressor" : mean_squared_error(y_test, prediction)**0.5,
                          "Linear Regressor" : mean_squared_error(y_test, preds2)**0.5
                          }, index = range(1)).T
benchmarks.columns=['RMSE']
benchmarks['Regressor'] = benchmarks.index
benchmarks.index = range(5)
benchmarks

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
# Any results you write to the current directory are saved as output.
print("Let's see a sample\n",df.sample(2))
print("Information\n",df.info())
df.drop(columns="Serial No.",inplace= True)
plt.hist(df['GRE Score'])
plt.hist(df['TOEFL Score'])
plt.hist(df['CGPA'])
df.isna().sum()
df.corr()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
train_x,test_x,train_y,test_y = train_test_split(df.iloc[:,:-1], df.iloc[:,-1])
lin_model = LinearRegression()
lin_model.fit(train_x,train_y)
target = lin_model.predict(test_x)
print("Mean Squared Error: ",mean_squared_error(test_y,target))

target[:5]
test_y[:5]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scale = scaler.fit(df.iloc[:,:-1])
scale_train = train_scale.transform(df.iloc[:,:-1])
train_x,test_x,train_y,test_y = train_test_split(scale_train, df.iloc[:,-1])
lin_model = LinearRegression()
lin_model.fit(train_x,train_y)
target = lin_model.predict(test_x)
print("Mean Squared Error: ",mean_squared_error(test_y,target))
from xgboost import XGBRegressor
model = XGBRegressor(max_depth = 6)
train_x,test_x,train_y,test_y = train_test_split(df.iloc[:,:-1], df.iloc[:,-1])
model.fit(train_x,train_y)
target = model.predict(test_x)
print("RMS ERROR: ",mean_squared_error(target,test_y))
print (train_y[:2])
plt.plot(range(len(test_y)),test_y)
plt.plot(range(len(target)),target)
df_sort = df.sort_values(by=df.columns[-1],ascending=False)
df_sort.head(5)

df_sort[(df_sort['Chance of Admit ']>0.9)].mean().reset_index()

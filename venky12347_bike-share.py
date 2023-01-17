# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
bike=pd.read_csv("../input/bike_share.csv")
bike.shape
bike.columns
bike.info()
bike.head(5).T
bike.tail(5).T
bike.isna().sum()
bike.isnull().apply(lambda x : [ sum(x), (sum(x) * 100) / bike.shape[0]] )
bike.describe().T
bike_corr=bike.corr()

bike_corr
bike_cov=bike.cov()

bike_cov
import matplotlib.pyplot as plt

plt.figure(figsize = (10,5))

ax = sns.boxplot(data = bike, orient = "h", color = "violet", palette = "Set1")

plt.show()
all_columns         = list(bike)

numeric_columns     = ['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered','count']

categorical_columns = [x for x in all_columns if x not in numeric_columns ]



print('\nNumeric columns')

print(numeric_columns)

print('\nCategorical columns')

print(categorical_columns)
def outlier_detect(bike):

    for i in bike.describe().columns:

        Q1=bike.describe().at['25%',i]

        Q3=bike.describe().at['75%',i]

        IQR=Q3 - Q1

        LTV=Q1 - 1.5 * IQR

        UTV=Q3 + 1.5 * IQR

        x=np.array(bike[i])

        p=[]

        for j in x:

            if j < LTV or j>UTV:

               p.append(j)

        print('\n Outliers for Column : ', i, ' Outliers count ', len(p))

        print(p)
x=bike[numeric_columns]

outlier_detect(x)
bike.nunique()
bike.duplicated().sum()
bike = bike.drop_duplicates()
bike.duplicated().sum()
bike.head(2)
plt.figure(figsize=(12,6))

g = sns.distplot(bike["registered"])

g.set_xlabel("registered", fontsize=12)

g.set_ylabel("Frequency", fontsize=12)

g.set_title("Frequency Distribuition- registered bikes", fontsize=20)
plt.figure(figsize=(12,6))

g = sns.distplot(bike["count"])

g.set_xlabel("count", fontsize=12)

g.set_ylabel("Frequency", fontsize=12)

g.set_title("Frequency Distribuition- bike count", fontsize=20)
bike.registered.plot(kind="hist")
bike.weather.value_counts().plot(kind="pie")
bike.season.value_counts().plot(kind="pie")
bike.groupby("weather").registered.agg(["mean","median","count"])
bike.groupby("season").registered.agg(["mean","median","count"])
bike.groupby("humidity").registered.agg(["mean","median","count"])
bike.groupby("temp").registered.agg(["mean","median","count"])
bike.groupby("atemp").registered.agg(["mean","median","count"])
# Set the width and height of the figure

plt.figure(figsize=(10,5))



corr = bike.corr()

ax = sns.heatmap(corr,vmin=-1,vmax=1,center=0,annot=True)
bike.head(5)
sns.scatterplot(x=bike['season'], y=bike['registered'])
plt.figure(figsize=(14, 7))

sns.scatterplot(x=bike['season'], y=bike['registered'],hue=bike['weather'],size=bike['temp'])
sns.pairplot(data=bike,hue='season')
sns.regplot(x=bike['temp'], y=bike['registered'])
sns.regplot(x=bike['humidity'], y=bike['registered'])
sns.regplot(x=bike['atemp'], y=bike['registered'])
sns.regplot(x=bike['casual'], y=bike['registered'])
%matplotlib inline

sns.pairplot(bike)
from sklearn.preprocessing import OneHotEncoder
# Importing necessary package for creating model

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
X = bike[['season','holiday','workingday','weather','temp','humidity','windspeed']]

y = bike['count']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)
model = LinearRegression()
model.fit(X_train,y_train)
model.coef_
model.intercept_
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
train_predict = model.predict(X_train)



mae_train = mean_absolute_error(y_train,train_predict)



mse_train = mean_squared_error(y_train,train_predict)



rmse_train = np.sqrt(mse_train)



r2_train = r2_score(y_train,train_predict)



mape_train = mean_absolute_percentage_error(y_train,train_predict)
test_predict = model.predict(X_test)



mae_test = mean_absolute_error(test_predict,y_test)



mse_test = mean_squared_error(test_predict,y_test)



rmse_test = np.sqrt(mean_squared_error(test_predict,y_test))



r2_test = r2_score(y_test,test_predict)



mape_test = mean_absolute_percentage_error(y_test,test_predict)
print('TRAIN: Mean Absolute Error(MAE): ',mae_train)

print('TRAIN: Mean Squared Error(MSE):',mse_train)

print('TRAIN: Root Mean Squared Error(RMSE):',rmse_train)

print('TRAIN: R square value:',r2_train)

print('TRAIN: Mean Absolute Percentage Error: ',mape_train)

print('TEST: Mean Absolute Error(MAE): ',mae_test)

print('TEST: Mean Squared Error(MSE):',mse_test)

print('TEST: Root Mean Squared Error(RMSE):',rmse_test)

print('TEST: R square value:',r2_test)

print('TEST: Mean Absolute Percentage Error: ',mape_test)
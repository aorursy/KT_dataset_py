import pandas as pd
df =pd.read_table('../input/auto-mpg-from-uci-site-directly/auto-mpg.data-original.txt', delim_whitespace=True, names=('mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name'))
df.info()
df['mpg'].median()
df['mpg'].fillna(value=df['mpg'].median(), inplace=True)
df.info()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
ls =df.columns.tolist()



df[ls].isnull().sum()
df["horsepower"].fillna(value=df["horsepower"].median(), inplace=True)  #same for "horsepower" column replacing nan with median


sns.pairplot(data=df)
df_feature = df[["displacement","horsepower","weight","acceleration"]]
df_target = df["mpg"]
plt.rcParams['figure.figsize']=(10,10)

plt.hist(df_target,bins=15)
import numpy as np



df_target_norm = np.log1p(df_target)
plt.rcParams['figure.figsize']=(10,10)

plt.hist(df_target_norm,bins=15)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
x_train_n,x_test_n,y_train_n,y_test_n = train_test_split(df_feature,df_target_norm, test_size=0.33, random_state=65)





x_train,x_test,y_train,y_test = train_test_split(df_feature,df_target, test_size=0.33, random_state=65)
lr = LinearRegression().fit(x_train,y_train)

lr_n = LinearRegression().fit(x_train_n,y_train_n)
lr.score(x_train,y_train)
lr.score(x_test,y_test)
lr_n.score(x_train_n,y_train_n)
lr_n.score(x_test_n,y_test_n)
from sklearn.metrics import mean_squared_error
y_pred = lr.predict(x_test)

y_pred_n = lr_n.predict(x_test_n)
from math import sqrt

rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_pred))

rmse_n = sqrt(mean_squared_error(y_true=y_test_n,y_pred=y_pred_n))



print("\n non log-transformed- %f \n log-transformed- %f" %(rmse,rmse_n))
plt.rcParams['figure.figsize']=(10,10)

plt.hist(y_pred-y_test)
plt.rcParams['figure.figsize']=(8,8)

plt.hlines(y=0,xmin=0,xmax=5000)

plt.scatter(x_test["weight"],y_pred-y_test)
plt.rcParams['figure.figsize']=(10,10)

plt.hist(y_pred_n-y_test_n)
plt.rcParams['figure.figsize']=(8,8)

plt.hlines(y=0,xmin=0,xmax=5000)

plt.scatter(x_test["weight"],y_pred_n-y_test_n)
df_features_norm = np.log1p(df_feature)
x2_train,x2_test,y2_train,y2_test = train_test_split(df_features_norm,df_target_norm, test_size=0.33, random_state=65)
lr2 = LinearRegression().fit(x2_train,y2_train)
lr2.score(x2_train,y2_train)
lr2.score(x2_test,y2_test)
lr2_y_pred = lr2.predict(x2_test)
rmse_lr2 = sqrt(mean_squared_error(y_true=y2_test,y_pred=lr2_y_pred))



print("\n %f" %(rmse_lr2))
plt.rcParams['figure.figsize']=(10,10)

plt.hist(lr2_y_pred-y2_test)
plt.rcParams['figure.figsize']=(8,8)

plt.hlines(y=0,xmin=6,xmax=10)

plt.scatter(x2_test["weight"],lr2_y_pred-y2_test)
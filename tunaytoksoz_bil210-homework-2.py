import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
startups = pd.read_csv("../input/50startups/50_Startups.csv", sep=",")

df=startups.copy()
df.head()
df.info()
df.shape
df.isnull().sum()
cor = df.corr()

cor
sns.set(rc={'figure.figsize':(10,3)})

sns.heatmap(cor, annot=True)
sns.set(rc={'figure.figsize':(10,5)})

sns.scatterplot("R&D Spend","Profit",data=df)
dsc = df.describe()

sns.set(rc={'figure.figsize':(10,5)})

plt.hist(dsc,bins=5)

plt.show()
dsc.T
df["State"].unique()
df_State = pd.get_dummies(df["State"])

df_State.head()
df_State.info()
df = pd.concat([df, df_State], axis = 1)

df.drop(["State","Florida"], axis = 1, inplace = True)
df.head()
X = df.drop("Profit", axis = 1)

y = df["Profit"]
X
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=45)
X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X,y)
y_pred = model.predict(X)

y_pred
df["tahminiProfit"] = y_pred

df
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

import math

MSE = mean_squared_error(df["Profit"],df["tahminiProfit"])

MAE = mean_absolute_error(df["Profit"],df["tahminiProfit"])

RMSE = math.sqrt(MSE)

print("MSE: ",MSE,"MAE:",MAE,"RMSE:",RMSE)
model.score(X,y)
import statsmodels.api as sm



stmodel = sm.OLS(y, X).fit()
stmodel.summary()
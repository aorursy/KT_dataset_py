#imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#load the data

df = pd.read_csv("../input/insurance.csv")
#head

df.head()
#describe

df.describe()
#info

df.info()
#nulls checking

df.isnull().any()
#data preperation

df["sex"] = df["sex"].replace("male","0").replace("female","1")

df["smoker"] = df["smoker"].replace("no","0").replace("yes","1")

df.head()
dfc = df.corr()

sns.heatmap(dfc,annot= True)
#charges linear model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X = df[["age","bmi","children","sex","smoker"]]

y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

mdl = LinearRegression()
mdl.fit(X_train,y_train)
mdl.score(X_test,y_test)
plt.scatter(mdl.predict(X_test),y_test),plt.scatter(mdl.predict(X_train),y_train)
plt.scatter(X_test["bmi"],y_test),plt.scatter(X_train["bmi"],y_train)
predictions = mdl.predict(X_test)

plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
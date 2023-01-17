
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Admission_Predict.csv')
df.head()
df.shape

df.describe()

df.dtypes
df.info()
df.isnull().sum()
df.head()
df['GRE Score'] = df['GRE Score']/df['GRE Score'].max()
df.head()
df['TOEFL Score'].describe()
df['TOEFL Score'] = df['TOEFL Score'] / df['TOEFL Score'].max()
df.head()
df['University Rating'].unique()
df['Research'].unique()
df['University Rating'].value_counts()
df['Research'].value_counts()
uni_rating_count = df['University Rating'].value_counts()
uni_rating_count
uni_rating_count.rename(columns={'University Rating':'value_counts'}, inplace=True)
uni_rating_count.index.name = 'University Rating'
uni_rating_count
df.head()
gre_score = df["GRE Score"]
chance_of_admit = df["Chance of Admit "]
plt.scatter(gre_score, chance_of_admit)
plt.title("Scatter plot of GRE Score vs Chance of Admission")
plt.xlabel("GRE Score")
plt.ylabel("Chance of Admission")
toefl_score = df["TOEFL Score"]
chance_of_admit = df["Chance of Admit "]
plt.scatter(toefl_score, chance_of_admit)
plt.title("Scatter plot of TOEFL Score vs Chance of Admission")
plt.xlabel("TOEFL Score")
plt.ylabel("Chance of Admission")
import seaborn as sns
sns.boxplot(x="University Rating", y="Chance of Admit ", data=df)
sns.boxplot(x="Research", y="Chance of Admit ", data=df)
df_grp = df[["University Rating", "Chance of Admit "]].groupby('University Rating', as_index=False).mean()
df_grp
cgpa = df["CGPA"]
chance_of_admit = df["Chance of Admit "]
plt.scatter(cgpa, chance_of_admit)
plt.title("Scatter plot of CGPA vs Chance of Admission")
plt.xlabel("CGPA")
plt.ylabel("Chance of Admission")
df_grp1 = df[["University Rating", "Research", "Chance of Admit "]].groupby(['University Rating', 'Research'], as_index=False).mean()
df_pivot_ru = df_grp1.pivot(index="Research", columns="University Rating")
df_pivot_ru
df.columns
sns.regplot(x="CGPA", y="Chance of Admit ", data=df)
plt.ylim(0,)
sns.regplot(x="GRE Score", y="Chance of Admit ", data=df)
plt.ylim(0,)
sns.regplot(x="TOEFL Score", y="Chance of Admit ", data=df)
plt.ylim(0,)
df.corr()
sns.lmplot(x='CGPA', y='Chance of Admit ',data=df)
sns.residplot(df['CGPA'], df['Chance of Admit '])
sns.residplot(df['GRE Score'], df['Chance of Admit '])
sns.residplot(df['TOEFL Score'], df['Chance of Admit '])
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'CGPA']]
y = df[['Chance of Admit ']]
y.shape
X.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print("Number of training data is ", x_train.shape[0])
print("Number of testing data is ", x_test.shape[0])
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm
x_train.shape, y_train.shape
lm.fit(x_train, y_train)
lm.intercept_
lm.coef_
Yhat = lm.predict(x_test)
Yhat[0:5]
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, Yhat)
lm.score(x_test, y_test)
df.head()
new_input = [[0.923529, 0.858333, 2, 8.21]]
lm.predict(new_input)
from sklearn.linear_model import Ridge
RidgeModel = Ridge(alpha=0.01)
RidgeModel.fit(x_train, y_train)
RidgeModel.coef_
RidgeModel.intercept_
Yhat_ridgeModel = RidgeModel.predict(x_test)
Yhat_ridgeModel[0:5]
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, Yhat_ridgeModel)
RidgeModel.score(x_test, y_test)
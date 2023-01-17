import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import cufflinks as cf
df = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
df.head()
df.info()
df.describe()
sns.pairplot(df, hue='Gender')
df["Spending Score (1-100)"].describe()
plt.figure(figsize=(12,6))

sns.distplot(df['Spending Score (1-100)'], kde=False,bins=30)

sns.despine()
plt.figure(figsize=(10,6))

sns.barplot(x='Gender',y='Spending Score (1-100)',data=df)

sns.despine()
plt.figure(figsize=(10,6))

sns.countplot(x='Gender',data=df,palette="vlag")

sns.despine()
plt.figure(figsize=(12,10))

sns.boxplot(x="Gender", y="Annual Income (k$)",data=df, palette="mako")
plt.figure(figsize=(12,10))

sns.boxplot(x="Gender", y="Spending Score (1-100)",data=df, palette="icefire")
x = df.corr()

x
plt.figure(figsize=(12,6))

sns.heatmap(x,cmap='coolwarm',annot=True,linecolor='white',linewidths=1)
plt.figure(figsize=(12,6))

plt.hist(df['Age'])

sns.despine()
df["Annual Income (k$)"].describe()
plt.figure(figsize=(8,5))

df['Annual Income (k$)'].plot.density()
plt.figure(figsize=(8,5))

df['Annual Income (k$)'].plot(kind='hist')

sns.despine()
sns.lmplot(x='Age', y='Annual Income (k$)', data=df, col='Gender', palette='pastel')
sns.lmplot(x='Age', y='Spending Score (1-100)', data=df, col='Gender', palette='pastel')
df.plot.line(x='Annual Income (k$)',y='Spending Score (1-100)',figsize=(12,3),lw=1)
df.plot.scatter(x='Annual Income (k$)',y='Spending Score (1-100)')
from sklearn.model_selection import train_test_split
df = pd.get_dummies(df, drop_first = True)

df.head()
X = df.drop('Spending Score (1-100)', axis=1)

y = df['Spending Score (1-100)']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=30);
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
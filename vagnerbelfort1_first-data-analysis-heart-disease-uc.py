import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/heart.csv")
df.head()
df.info()
df.shape
df.isnull().sum()
df.describe()
plt.figure(figsize=(14,14))

sns.heatmap(df.corr(), annot = True, linewidth = 0.1, cmap='coolwarm')
df.target.value_counts()
sns.countplot(x="target",data=df)
sns.countplot(x='sex', data=df)

plt.xlabel("Sex (0 = homem, 1= mulher)")

plt.show()
df['age'].hist(bins=30)

plt.xlabel('Age')
plot = df[df.target == 1].age.value_counts().sort_index().plot(kind="bar", figsize=(15,4))

plot.set_title("Age distribution")
plot = df[df.target == 0].age.value_counts().sort_index().plot(kind="bar", figsize=(15,4))

plot.set_title("Age distribution")
df.cp.value_counts()
sns.countplot(x="cp",data=df)
sns.countplot(x='target',hue='cp',data=df)
pd.crosstab(df.cp, df.target).plot(kind='bar')

plt.title("Frequência de doença cardíaca em relação ao tipo de dor")

plt.xlabel("Tipo de dor ")

plt.ylabel("tem a doença ou nao")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis=1), df['target'], test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
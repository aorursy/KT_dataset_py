import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_iris = pd.read_csv('../input/Iris.csv')
df_iris.info()
df_iris.head()
df_iris=df_iris.drop(['Id'], axis=1)
df_iris.describe()
sns.set(style='ticks')

sns.pairplot(df_iris.dropna(), hue='Species')

plt.show()
plt.figure(figsize=(10, 10))

for column_index, column in enumerate(df_iris.columns):

    if column == 'Species':

        continue

    plt.subplot(2, 2, column_index + 1)

    sns.violinplot(x='Species', y=column, data=df_iris )

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

X = df_iris.drop(['Species'],axis=1)

Y = df_iris['Species']

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, train_size=0.75, random_state=1)

alg = DecisionTreeClassifier()

alg.fit(X_train, Y_train)
Y_pred = alg.predict(X_test)

print(Y_pred)
scores = accuracy_score(Y_test, Y_pred)

print(scores)
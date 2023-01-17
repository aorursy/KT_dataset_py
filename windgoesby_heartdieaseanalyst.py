# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
df.info()
df.target.value_counts()
sns.countplot(x='target', data=df, palette='bwr')

plt.xlabel("Target (0 = diagnose, 1= no diagnose)")

plt.show()
sns.countplot(x='sex', data=df, palette='mako_r')

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()
# Patient age distribution

sns.violinplot(x=df[df['target']==1]['sex'],y=df[df['target']==1]['age'], palette='pastel')

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()
df.groupby('target').mean()
pd.crosstab(df.sex, df.target).plot(kind='bar')

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()
df.head()
df.thal.value_counts()
# one-hot encoder

cp = pd.get_dummies(df['cp'], prefix='cp')

ca = pd.get_dummies(df['ca'], prefix='ca')

thal = pd.get_dummies(df['thal'], prefix='thal')

df = pd.concat([df, cp, ca, thal], axis=1)

df.head()
df = df.drop(['cp', 'ca', 'thal'], axis=1)

df.head()
label = df['target']

features = df.drop('target', axis=1)
# MinMax

features = (features - np.min(features)) / (np.max(features) - np.min(features)).values
x_train, x_test, y_train, y_test = train_test_split(features,label,test_size = 0.2,random_state=0)
lr = LogisticRegression()

lr.fit(x_train, y_train)

predictions = lr.predict(x_test)

print(lr.score(x_test,y_test)*100)
acc_results = []

for i in range(1, 20):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train, y_train)

    predictions = knn.predict(x_test)

#     print(knn.score(x_test, y_test)*100)

    acc_results.append(knn.score(x_test, y_test)*100)

plt.plot(range(1, 20), acc_results)

print(max(acc_results))
svm = SVC(random_state=1)

svm.fit(x_train, y_train)

print(svm.score(x_test, y_test)*100)
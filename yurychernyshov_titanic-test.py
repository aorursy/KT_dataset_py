# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df
df.shape
df.info()
df.describe()
df.Survived.unique(), df.Survived.value_counts()
df.Pclass.unique(), df.Pclass.value_counts()
df.Sex.unique(), df.Sex.value_counts()
df.Age.isna().sum()
df1 = df[~df.Age.isna()]
plt.boxplot(df1.Age)
df.Age.min(), df.Age.max()
df[df.Age==df.Age.min()]
df.groupby(['Pclass', 'Sex']).PassengerId.count()
df.groupby(['Pclass', 'Sex', 'Survived']).PassengerId.count()
plt.scatter(df.Fare, df.Age)
df[df.Fare>500]
df1 = df[df.Fare<100]
plt.scatter(df1.Fare, df1.Age)
df.Age.isna().sum()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
df.Age.isna().sum()
df2 = df[~df.Age.isna()]
X = df2.Fare.values
y = df2.Age
X.shape, y.shape
model.fit(X.reshape(-1,1), y)
result = model.predict(X.reshape(-1,1))
plt.figure(figsize=(20,10))
X1 = df[df.Age.isna()].Fare
plt.scatter(df2.Fare, df2.Age, c='b')
plt.plot(df2.Fare, result, c='r')
plt.scatter(X1, model.predict(np.array(X1).reshape(-1,1)), s=50, c='g')
dft = pd.read_csv('/kaggle/input/titanic/test.csv')
dft
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
X = df2[['Age', 'Fare']]
y = df2.Survived
clf.fit(X, y)
dft.Age.fillna(dft.Age.mean(), inplace=True)
dft.Fare.fillna(dft.Age.mean(), inplace=True)
X_test = dft[['Age','Fare']]
result = clf.predict(X_test)
result
answer = pd.DataFrame({'PassengerId':dft.PassengerId, 'Survived':result})
answer.to_csv('submission.csv', index=False)


#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import math

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.head()
df_test.head()
df_train.isnull().sum()
df_train.dtypes
df_train.Survived = df_train.Survived.astype('category')

df_train.Pclass = df_train.Pclass.astype('category')

df_train.SibSp = df_train.SibSp.astype('category')

df_train.Parch = df_train.Parch.astype('category')
sns.countplot('Survived',data=df_train)

plt.show()
df_train = df_train.drop(columns="PassengerId")

df_train = df_train.drop(columns="Cabin")
df_train.fillna(df_train.mean(), inplace=True)

corrmat = df_train.corr().abs()
plt.ylim(0, 100)

sns.boxplot(data=df_train,  y="Fare", x = "Survived")



plt.ylim(0, 60)

sns.boxplot(data=df_train,  y="Age", x = "Survived")

pd.crosstab(df_train.Survived, df_train.Pclass)
pd.crosstab(df_train.Survived, df_train.Parch)

pd.crosstab(df_train.Survived, df_train.SibSp)
pd.crosstab(df_train.Survived, df_train.Sex)
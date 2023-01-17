#Importing libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os



%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings('ignore')
#Reading the Data



df = pd.read_csv("/kaggle/input/Heart.csv")
df.shape
df.head()
#Checking Null values



df.isnull().sum()
df.target.value_counts()
sns.countplot(df.target)

plt.show()
df.groupby(["target", "sex"])["sex"].count()
sns.countplot(df.target, hue=df.sex)
sns.distplot(df.thalach)
df['target'].corr(df['thalach'])
plt.figure(figsize=(15,8))

sns.heatmap(df.corr(), annot = True)

plt.show()
df.age.describe()
sns.distplot(df.age)

plt.show()
df['target'].corr(df['age'])


plt.pie(df.groupby(["cp"])["cp"].count(), labels = df["cp"].unique()

)

plt.show()
sns.scatterplot(df.target, df.thalach)
df.thalach[df['target']==0 & df["thalach"]].max()
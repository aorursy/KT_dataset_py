import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
tips = pd.read_csv('tips.csv')  # import dataset

tips  # look at the data
tips.head()
tips.tail()
tips.describe()
tips.describe(include=['O'])
tips.isnull()
tips.isnull().any(axis=0)
tips.isnull().any(axis=1).head(13)
tips.isnull().sum(axis=0)
tips.isnull().sum(axis=0).sum()
tips.shape

# the size of the dataset (rows, cols)
tips.isnull().sum(axis=0) / tips.shape[0]
sns.relplot(x="total_bill", y="tip", data=tips)
sns.relplot(x="total_bill", y="tip", col="time", data=tips)
sns.relplot(x='total_bill', y='tip', hue='sex', data=tips)
sns.relplot(x='total_bill', y='tip', hue='smoker', style='smoker', data=tips)
sns.relplot(x='total_bill', y='tip', size='size', data=tips)
sns.relplot(x='total_bill', y='tip', col='time',

            hue='smoker', style='sex', size='size', data=tips)
sns.distplot(tips.total_bill, kde=False)
sns.distplot(tips.total_bill, kde=True)
sns.distplot(tips.total_bill, bins=20, kde=True)
sns.distplot(tips.total_bill, bins=20, kde=True, rug=True)
sns.distplot(tips['tip'], bins=12, kde=True, rug=True)
sns.jointplot(x="total_bill", y="tip", data=tips)
sns.pairplot(tips)
sns.catplot(x="time", y="total_bill", data=tips)
sns.catplot(x="time", y="total_bill", hue='smoker', data=tips)
sns.catplot(x="time", y="total_bill", hue='smoker', dodge=True, data=tips)
sns.catplot(x="time", y="total_bill", hue='sex',

            dodge=True, kind='swarm', data=tips)
sns.catplot(x="time", y="total_bill", hue='time',

            dodge=False, kind='bar', data=tips)
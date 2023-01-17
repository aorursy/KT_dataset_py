# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
titanic_data = pd.read_csv("../input/train.csv")
null_data = titanic_data.columns[titanic_data.isnull().any()]

null_data
titanic_data.dtypes
fig = plt.figure(figsize=(8, 8))

ax = fig.gca()

titanic_data.plot(kind='scatter', x='Age', y='Survived', ax=ax)

ax.set_title("Scatter plot Pclass vs Fare")

ax.set_xlabel('Age')

ax.set_ylabel('Survived')
counts = titanic_data['Survived'].value_counts()

counts
fig = plt.figure(figsize=(8, 8))

ax = fig.gca()

counts.plot.bar(ax=ax)

ax.set_title("Number of Passengers Survived")

plt.xlabel('Survived')

plt.ylabel('Counts')
count = titanic_data['Pclass'].value_counts()

count
fig = plt.figure(figsize=(8, 8))

ax = fig.gca()

count.plot.bar(ax=ax)

ax.set_title("Number of Passengers per Passenger Class")

plt.xlabel('Passenger Class')

plt.ylabel('Counts')
fig = plt.figure(figsize=(8, 6))

ax = fig.gca()

titanic_data[['Age', 'Survived']].boxplot(by='Survived', ax=ax)

ax.set_title("BoxPlot of Age and Survived")

plt.xlabel('Survived')

plt.ylabel('Age')
fig = plt.figure(figsize=(8, 6))

ax = fig.gca()

titanic_data[['Fare', 'Pclass']].boxplot(by='Pclass', ax=ax)

ax.set_title("BoxPlot of Fare and Passenger Class")

plt.ylabel('Fare')

plt.xlabel('Pclass')
fig = plt.figure(figsize=(10, 10))

ax = fig.gca()

sns.set_style("whitegrid")

sns.violinplot(x='Pclass', y='Survived', data=titanic_data, ax=ax)

ax.set_title("Violin plot of Passengers survived from each class")

plt.ylabel('Survived')

plt.xlabel('Pclass')
eCounts = titanic_data['Embarked'].value_counts()

eCounts
fig = plt.figure(figsize=(10, 10))

ax = fig.gca()

sns.set_style("whitegrid")

sns.violinplot(x='Embarked', y='Survived', data=titanic_data, ax=ax)

ax.set_title("Violin plot of Passengers embarked and Survived")

plt.ylabel('Survived')

plt.xlabel('Embarked')
fig = plt.figure(figsize=(10, 10))

ax = fig.gca()

sns.set_style("whitegrid")

sns.violinplot(x='Embarked', y='Pclass', data=titanic_data, ax=ax)

ax.set_title("Violin plot of Passengers embarked")

plt.ylabel('Pclass')

plt.xlabel('Embarked')
pcounts = titanic_data['Pclass'].value_counts()

pcounts
fig = plt.figure(figsize=(10, 10))

ax = fig.gca()

sns.set_style("whitegrid")

sns.violinplot(x='Sex', y='Pclass', data=titanic_data, ax=ax)

ax.set_title("Violin plot of Sex and Pclass")

plt.ylabel('Pclass')

plt.xlabel('Sex')
fig = plt.figure(figsize=(10, 10))

ax = fig.gca()

sns.set_style("whitegrid")

sns.violinplot(x='Sex', y='Survived', data=titanic_data, ax=ax)

ax.set_title("Violin plot of Passengers embarked")

plt.ylabel('Survived')

plt.xlabel('Sex')
sns.lmplot(x='Age', y='Pclass', data=titanic_data, hue='Survived',

           palette='Set2', fit_reg=False, scatter_kws={"alpha": 0.3}, size=7)
sns.lmplot(x='Age', y='Pclass', data=titanic_data, hue='Embarked',

           palette='Set2', fit_reg=False, scatter_kws={"alpha": 0.3}, size=7)
fig = plt.figure(figsize=(10, 8))

ax = fig.gca()

sns.set_style("whitegrid")

sns.violinplot(data=titanic_data, x='Embarked', y='Pclass', hue='Survived', split=True, ax=ax)
fig = plt.figure(figsize=(10, 8))

ax = fig.gca()

sns.set_style("whitegrid")

sns.violinplot(data=titanic_data, x='Survived', y='Pclass', hue='Sex', split=True, ax=ax)
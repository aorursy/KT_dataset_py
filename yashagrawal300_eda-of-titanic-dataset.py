# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import os

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
test = pd.read_csv('/kaggle/input/titanic/test.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.shape
train.head()
train.isnull().sum()
sns.countplot('Survived', data = train)
train.groupby(['Sex', 'Survived'])['Survived'].count()

sns.countplot('Sex',hue = 'Survived', data = train)
train.groupby(['Pclass', 'Survived'])['Survived'].count()
sns.countplot('Pclass', hue = 'Survived', data = train)
train['Age'].describe()
train['Age']=train['Age'].fillna(train['Age'].mean())

train.isnull().sum()
sns.swarmplot(train['Survived'],train['Age'])
sns.swarmplot(train['Survived'],train['Age'], hue = 'Sex', data = train, split = True)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import pandas_profiling



train = pd.read_csv('../input/titanic/train.csv')

train.profile_report()
import matplotlib.pyplot as plt

import seaborn as sns
# 死亡者

plt.hist(train.loc[train['Survived'] == 0, 'Age'].dropna(), bins=30, alpha=0.5, label='0')

# 生存者

plt.hist(train.loc[train['Survived'] == 1, 'Age'].dropna(), bins=30, alpha=0.5, label='1')



plt.xlabel('Age')

plt.ylabel('count')

plt.legend(title = 'Surivied')

plt.show()
# SibSp と Survived の件数を棒グラフ

sns.countplot(x='SibSp', hue='Survived', data=train)

plt.legend(loc='upper right', title='Survived')

plt.show()
sns.countplot(x='Parch', hue='Survived', data=train)

plt.legend(loc='upper right', title='Survived')

plt.show()
plt.hist(train.loc[train['Survived'] == 0, 'Fare'].dropna(), range=(0, 250), bins=25, alpha=0.5, label='0')

plt.hist(train.loc[train['Survived'] == 1, 'Fare'].dropna(), range=(0, 250), bins=25, alpha=0.5, label='1')

plt.xlabel('Fare')

plt.ylabel('Survived')

plt.legend(title='Survived')

plt.xlim(-5, 250)

plt.show()
sns.countplot(x='Pclass', hue='Survived', data=train)

plt.show()
sns.countplot(x='Sex', hue='Survived', data=train)

plt.show()
sns.countplot(x='Embarked', hue='Survived', data=train)

plt.show()
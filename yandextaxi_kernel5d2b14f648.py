# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

fname = '/kaggle/input/titanic/train.csv'

data = pd.read_csv(fname)
len(data)
data.head()
data.count()
data['Age'].min(), data['Age'].max()
data['Survived'].value_counts()
data['Survived'].value_counts() * 100 / len(data)
data['Sex'].value_counts()
data['Pclass'].value_counts()
%matplotlib inline

alpha_color = 0.5

data['Survived'].value_counts().plot(kind='bar')
data['Sex'].value_counts().plot(kind='bar',

                               color=['b', 'r'],

                               alpha=alpha_color)
data['Pclass'].value_counts().sort_index().plot(kind='bar',

                                               alpha=alpha_color)
data.plot(kind='scatter', x = 'Survived',y='Age')
data[data['Survived'] == 1]['Age'].value_counts().sort_index().plot(kind='bar')
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]

data['AgeBin'] = pd.cut(data['Age'], bins)
data[data['Survived'] == 1]['AgeBin'].value_counts().sort_index().plot(kind='bar')
data[data['Survived'] == 0]['AgeBin'].value_counts().sort_index().plot(kind='bar')
data['AgeBin'].value_counts().sort_index().plot(kind='bar')
data['AgeBin'].value_counts().sort_index().plot(kind='bar')
data[data['Pclass'] == 1]['Survived'].value_counts().plot(kind='bar')
data[data['Pclass'] == 3]['Survived'].value_counts().plot(kind='bar')
data[data['Sex'] == 'male']['Survived'].value_counts().sort_index().plot(kind='bar')
data[data['Sex'] == 'female']['Survived'].value_counts().plot(kind='bar')
data[(data['Sex'] == 'male') & (data['Pclass'] == 1)]['Survived'].value_counts().sort_index().plot(kind='bar')
data[(data['Sex'] == 'male') & (data['Pclass'] == 3)]['Survived'].value_counts().plot(kind='bar')
data[(data['Sex'] == 'female') & (data['Pclass'] == 1)]['Survived'].value_counts().plot(kind='bar')
data[(data['Sex'] == 'female') & (data['Pclass'] == 3)]['Survived'].value_counts().plot(kind='bar')
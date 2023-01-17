# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/titanic-data/train.csv')
data.head
data.info
sns.heatmap(data.isnull(), cbar=False)
data.drop('Cabin', axis=1, inplace=True)
data.head
# Filling Nan

data['Age'].fillna(0, inplace=True)
# Replacing sex

data['Sex'].replace(['female','male'], [0, 1], inplace=True)
# Replacing embarked

data['Embarked'].replace(['S','C', 'Q'], [1, 2, 3], inplace=True)
data
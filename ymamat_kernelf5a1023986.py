# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
sns.regplot(data=train,
           x='GrLivArea',
           y='SalePrice'
          )
sns.distplot(train['SalePrice'], bins=20)
train.columns
sns.lmplot(data=train,x='YearBuilt', y='SalePrice', row='BldgType')

sns.countplot(train['Neighborhood'])
plt.xticks(rotation='vertical')

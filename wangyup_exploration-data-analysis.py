# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_mat = pd.read_csv('../input/train.csv')

test_mat = pd.read_csv('../input/test.csv')

macro_mat = pd.read_csv('../input/macro.csv')

train_mat = pd.merge(left=train_mat, right=macro_mat, how='left', on='timestamp')

test_mat = pd.merge(left=test_mat, right=macro_mat, how='left', on='timestamp')
missing_values = pd.DataFrame(train_mat.isnull().astype(np.int64).sum().reset_index())

missing_values.columns = ['fea_name', 'missing_count']

missing_values = missing_values[missing_values.missing_count > 0]

missing_values.ix[:, 'pre'] = missing_values.missing_count.apply(lambda x: float(x) / len(train_mat))
missing_values = missing_values.sort_values(by='missing_count')

missing_values
sns.barplot(x='missing_count', y='fea_name', data=missing_values)
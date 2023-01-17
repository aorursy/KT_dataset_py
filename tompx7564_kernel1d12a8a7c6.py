# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import pandas as pd #Analysis 

import matplotlib.pyplot as plt #Visulization

import seaborn as sns #Visulization

import numpy as np #Analysis 

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats #Analysis 

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
print('train.csv Shape : ',df_train.shape)

print('test.csv Shape : ',df_test.shape)
df_train.head()
df_train.describe()
df_test.describe()
for col in df_train.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)
for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)
f, ax = plt.subplots(figsize=(8,6))

sns.distplot(df_train['price'])
print("Skewness : %f" % df_train['price'].skew())

print('Kutosis : %f'% df_train['price'].kurt())
fig = plt.figure(figsize = (15,10))

fig.add_subplot(1,2,1)

res = stats.probplot(df_train['price'],plot=plt)



fig.add_subplot(1,2,2)

res = stats.probplot(np.log1p(df_train['price']),plot=plt)
df_train['price'] = np.log1p(df_train['price'])

#histogram

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['price'])
k = 10

corrmat = abs(df_train.corr(method='spearman'))

cols = corrmat.nlargest(k,'price').index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

f,ax = plt.subplots(figsize=(18,8))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='grade', y="price", data=data)
data = pd.concat([df_train['price'], df_train['sqft_living']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='sqft_living', y="price", data=data)
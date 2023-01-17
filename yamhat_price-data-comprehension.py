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

import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_formats = {'png', 'retina'}
import seaborn as sns
train = pd.read_csv('../input/property-price-prediction-challenge-2nd/DC_train.csv')
test = pd.read_csv('../input/property-price-prediction-challenge-2nd/DC_test.csv')
print('train shape : ', train.shape)
print('test shape : ', test.shape)
train.describe()
fig, ax = plt.subplots(figsize=(14, 6))
plt.hist(train['PRICE'], bins=50, rwidth=0.8)
plt.xlabel('PRICE')
plt.ylabel('Frequency')
plt.show()
fig, ax = plt.subplots(figsize=(14, 6))
plt.hist(train['PRICE'], bins=50, rwidth=0.8)
plt.xlim(750000, 25000000)
plt.ylim(0, 100)
plt.xlabel('PRICE')
plt.ylabel('Frequency')
plt.show()
fig, ax = plt.subplots(figsize=(14, 6))
sns.set_style('white')
sns.distplot(train['PRICE'], color='blue')
plt.xlabel('PRICE')
plt.show()
train['LOGPRICE'] = np.log1p(train['PRICE'])
train['LOGPRICE'].describe()
fig, ax = plt.subplots(figsize=(14, 6))
sns.set_style('white')
sns.distplot(train['LOGPRICE'], color='blue')
plt.xlabel('LOGPRICE')
plt.show()
print('Before deleting outlier')
print('Skewness : ', train['LOGPRICE'].skew())
print('Kurtosis : ', train['LOGPRICE'].kurt())
mean = np.mean(train['LOGPRICE'])
var = np.var(train['LOGPRICE'])
print('Mean : {:.2f}'.format(mean))
print('Variance : {:.2f}'.format(var))

train = train[(train['LOGPRICE']<=17.0) & (train['LOGPRICE']>=8.0)]
fig, ax = plt.subplots(figsize=(14, 6))
sns.distplot(train['LOGPRICE'], color='blue')
plt.xlabel('LOGPRICE')
plt.show()
print('Before deleting outlier')
print('Skewness : ', train['LOGPRICE'].skew())
print('Kurtosis : ', train['LOGPRICE'].kurt())
mean = np.mean(train['LOGPRICE'])
var = np.var(train['LOGPRICE'])
print('Mean : {:.2f}'.format(mean))
print('Variance : {:.2f}'.format(var))

train['LOGPRICE'].describe()
train['PRICE'].describe()
train.shape

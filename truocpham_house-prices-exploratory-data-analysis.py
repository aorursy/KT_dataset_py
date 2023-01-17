# data manipulation
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

# plotting
import seaborn as sn
import matplotlib.pyplot as plt
%matplotlib inline

# setting params
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

sn.set_style('whitegrid')
sn.set_context('talk')

plt.rcParams.update(params)

# config for show max number of output lines
pd.options.display.max_colwidth = 600
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

# pandas display data frames as tables
from IPython.display import display, HTML

# modeling utilities
import scipy.stats as stats
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

import warnings
warnings.filterwarnings('ignore')
# Load train & test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Display dimension of datasets
print('Train data shape', train.shape)
print('Test data shape', test.shape)
train.head()
test.head()
# dataset summary stats
train.describe()
# data types of attributes of train set
train.dtypes
# Count unique missing value of each column
for col in train.columns:
    if train[col].isnull().values.any():
        print(col)
        print(train[col].isnull().sum())
total_missing = train.isnull().sum().sort_values(ascending=False)
ratio_missing = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_missing, ratio_missing], axis=1, keys=['Total', 'Ratio'])
missing_data['Type'] = train[missing_data.index].dtypes

missing_data = missing_data[(missing_data['Total'] > 0)]

# display missing data
missing_data
print('Numerical Missing Values:')
print('=========================')
[print(col_missing,  '\t', missing_data['Total'][col_missing], 'NaNs') \
 for col_missing in missing_data[(missing_data['Total'] > 0) & \
                                 (missing_data['Type'] != 'object')].index.values]
print('=========================')
print('Categorical Missing Values:')
print('=========================')
[print(col_missing,  '\t', missing_data['Total'][col_missing], 'NaNs') \
 for col_missing in missing_data[(missing_data['Total'] > 0) & \
                                 (missing_data['Type'] == 'object')].index.values]
print('=========================')
train['SalePrice'].describe()
train['SalePrice'].skew()
train['SalePrice'].kurt()
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.hist(train['SalePrice'], color='blue')
plt.xlabel('SalePrice')
plt.show()
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.hist(np.log1p(train['SalePrice']), color='blue')
plt.xlabel('SalePrice')
plt.show()
corr = train.select_dtypes(include=['float64', 'int64']).iloc[:,1:].corr()
f, ax = plt.subplots(figsize=(22, 22))
sn.heatmap(corr, vmax=.8, square=True)
# Correlation between attributes with SalePrice
corr_list = corr['SalePrice'].sort_values(axis=0, ascending=False).iloc[1:]
corr_list
# Scatter plotting the top related to SalePrice
plt.figure(figsize=(22, 22))
k = 6

for i in range(k):
    ii = '32'+str(i)
    plt.subplot(ii)
    feature = corr_list.index.values[i]
    plt.scatter(train[feature], train['SalePrice'], facecolors='none', edgecolors='k', s=75)
    sn.regplot(x=feature, y='SalePrice', data=train, scatter=False, color='b')
    ax=plt.gca()
    ax.set_ylim([0,800000])
# Scatter plotting the variables most correlated with SalePrice
cols = corr.nlargest(10, 'SalePrice')['SalePrice'].index
sn.set()
sn.pairplot(train[cols], size=2.5)
plt.show()
cat_df = train.select_dtypes(include=['object'])
cat_df.shape
cat_df.dtypes
for cat in cat_df.dtypes[:15].index.values:
    plt.figure(figsize=(16, 22))
    plt.xticks(rotation=90)
    sn.boxplot(x=cat, y='SalePrice', data=train)    
    plt.show()
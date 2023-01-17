import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb
color = sns.color_palette()
%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print('train rows {} and cols {}'.format(train_df.shape[0],train_df.shape[1]))
print('test rows {} and cols {}'.format(test_df.shape[0],test_df.shape[1]))
dtype_counts = train_df.get_dtype_counts()
print('Train dtype counts \n',dtype_counts)
dtype_columns = train_df.drop('target',axis=1).dtypes.reset_index()
print('---------------------\n')
print('Test info \n',test_df.info())
''' If you look closely the datatype of all columns of test_df is float64 except ID.
I first tried to merge the train and test data and then typecast the int_cols to int64
but that line of code got stuck for about 20 minutes.So i stopped the kernel and decided to typecast test_data separately 
and then concat them.'''
'''Also passing the dictionary to datatype when reading test.csv doesnt work'''
dtype_columns.columns = ['Cols','datatype'] ##change column names
int_cols = dtype_columns.Cols.loc[dtype_columns.datatype=='int64'].values.copy() 
Test_IntCols = test_df[int_cols].astype('int64',copy=False).copy() ##new view of data
test_df.drop(int_cols,axis=1,inplace=True) ## drop the columns in new view
test_df = test_df.merge(Test_IntCols,left_index=True,right_index=True) ##took around 3 seconds!! 
print(test_df.info()) ##sucessful conversion of datatypes
target = train_df.target.values.copy()
train_df.drop('target',axis=1,inplace=True)
print(train_df.info())
## column datatypes matched!!
test_df = pd.concat((train_df,test_df),axis=0) ##around 8 seconds!! HURRY!
test_df.info(memory_usage='deep')
##checking the missing values - should have done before merging i think
missing_values_df = test_df.isnull().sum(axis=0).reset_index()
missing_values_df.columns = ['cols','count']
missing_df = missing_values_df.loc[missing_values_df['count'] > 0]
print(missing_df)
unique_col_df = test_df.nunique().reset_index()
unique_col_df.columns = ['cols','unique_values']
# one_unique_element_cols = unique_col_df.loc[unique_col_df.unique_values==1]
unique_col_df = unique_col_df.sort_values(by='unique_values')
print(unique_col_df.head())
## Hmm! I dont believe it. The other kernels showed that train data has 256 columns which contains only 1 categorical element!
## Let's check the train data
unique_elem_col = test_df[int_cols].iloc[:target.__len__()].nunique().reset_index()
unique_elem_col.columns = ['cols','count']
unique_elem_col = unique_elem_col.sort_values(by='count').reset_index()
print(unique_elem_col.head())
test_df['eb6bb7ce1'].unique()
## taken from @srk's kernel
plt.figure(figsize=(8,6))
plt.scatter(range(target.__len__()),sorted(target))
plt.xlabel('Index')
plt.ylabel('Transaction value')
plt.tight_layout()
plt.show()

## exponential target values - this may be the reason for choosing RMSLE metric

##histogram plot
plt.figure(figsize=(8,6))
sns.distplot(target,bins=50,kde=False)
plt.xlabel('Transactions')
plt.ylabel('count')
plt.title('Target Histogram')

## This is right skewed Distribution
## Let's try log here
##histogram plot
plt.figure(figsize=(8,6))
sns.distplot(np.log1p(target),bins=50,kde=False)
plt.xlabel('Transactions')
plt.ylabel('count')
plt.title('Target Histogram')
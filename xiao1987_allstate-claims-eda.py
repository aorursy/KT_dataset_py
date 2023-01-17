import numpy as np

import pandas as pd 

import time

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns
train = pd.read_csv('../input/train.csv')

train_size = train.shape[0]

test = pd.read_csv('../input/test.csv')

test_size = test.shape[0]
train.shape
#Print all rows and columns. Dont hide any

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
print(train.head())
train.columns.values
train.isnull().sum()
# merge train & test to save time

full = pd.concat([train,test]).reset_index(drop=True)
train.dtypes
data_types = full.dtypes

cat_cols = list(data_types[data_types=='object'].index)

con_cols = list(data_types[data_types=='int64'].index)+list(data_types[data_types=='float64'].index)



id_col = 'id'

target_col = 'loss' 

con_cols.remove('id')

con_cols.remove('loss')



print('Categorical features:',cat_cols)

print('Numerical features:',con_cols)

print('ID:%s, target:%s'%(id_col,target_col))

#skew 

full.skew()

# loss shows the highest skew
#Visualize all the continuous data with violin plot

n_cols =2

n_rows =7



for i in range(n_rows):

    fg,ax = plt.subplots(nrows=1,ncols=n_cols,figsize=(12,8))

    for j in range(n_cols):

        sns.violinplot(y=con_cols[i*n_cols+j],data=train,ax=ax[j])
sns.violinplot(data=train,y=train['loss'])

plt.show()
#data transformation for 'loss'

#log1p func. applies log(1+x)to all elements of the column 

loss = np.log1p(train['loss'])
sns.violinplot(data=train,y=loss)

plt.show()
# explore correlations 

corrmat = train[con_cols].corr()

f,ax=plt.subplots(figsize=(12,9))

sns.heatmap(corrmat,vmax=.8,square=True)

plt.show()
# explore catagorical data by visualization 

n_cols = 4

n_rows = 29

for i in range(n_rows):

    fg,ax = plt.subplots(nrows=1,ncols=n_cols,figsize=(12,8))

    for j in range(n_cols):

        sns.countplot(x=cat_cols[i*n_cols+j],data=train,ax=ax[j])
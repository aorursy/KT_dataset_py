# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv('../input/train.csv')
print(train_data.shape)
train_data.head()
test_data = pd.read_csv('../input/test.csv')
print(test_data.shape)
test_data.head()
train_data['Target'].value_counts()
train_data['Target'].plot.hist()
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
missing_values = missing_values_table(train_data)
missing_values.head()
train_data.dtypes.value_counts()
train_data.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
train_data_1 = train_data.drop(['Id', 'idhogar'], axis=1)
test_data_1 = test_data.drop(['Id', 'idhogar'], axis=1)
train_data_1 = pd.get_dummies(train_data_1)
test_data_1 = pd.get_dummies(test_data_1)
print(train_data_1.shape)
print(test_data_1.shape)
train_label = train_data_1['Target']
train_data_1, test_data_1 = train_data_1.align(test_data_1, join='inner', axis=1)
train_data_1['Target'] = train_label
print(train_data_1.shape)
print(test_data_1.shape)
correlations = train_data_1.corr()['Target'].sort_values()
print(correlations.tail())
print(correlations.head())
# 读书时间越少，越贫穷
plt.figure(figsize=(10, 8))
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 1, 'escolari'], label='target==1')
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 2, 'escolari'], label='target==2')
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 3, 'escolari'], label='target==3')
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 4, 'escolari'], label='target==4')
# 孩子越多越贫穷
plt.figure(figsize=(10, 8))
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 1, 'hogar_nin'], label='target==1')
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 2, 'hogar_nin'], label='target==2')
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 3, 'hogar_nin'], label='target==3')
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 4, 'hogar_nin'], label='target==4')
plt.figure(figsize=(10, 8))
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 1, 'r4t1'], label='target==1')
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 2, 'r4t1'], label='target==2')
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 3, 'r4t1'], label='target==3')
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 4, 'r4t1'], label='target==4')
plt.figure(figsize=(10, 8))
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 1, 'meaneduc'], label='target==1')
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 2, 'meaneduc'], label='target==2')
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 3, 'meaneduc'], label='target==3')
sns.kdeplot(train_data_1.loc[train_data_1['Target'] == 4, 'meaneduc'], label='target==4')
from sklearn.preprocessing import MinMaxScaler, Imputer
if 'Target' in train_data_1:
    train_data_1 = train_data_1.drop(columns=['Target'])
else:
    train_data_1 = train_data_1.copy()

features = list(train_data_1.columns)
imputer = Imputer(strategy='median')
scaler = MinMaxScaler(feature_range=(0, 1))

imputer.fit(train_data_1)
train_data_1 = imputer.transform(train_data_1)
test_data_1 = imputer.transform(test_data_1)

scaler.fit(train_data_1)
train_data_1 = scaler.transform(train_data_1)
test_data_1 = scaler.transform(test_data_1)

print(train_data_1.shape)
print(test_data_1.shape)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
rf.fit(train_data_1, train_label)
predictions = rf.predict(test_data_1)
submit = pd.read_csv('../input/sample_submission.csv')
submit['Target'] = predictions
submit.to_csv('rf_baseline.csv', index=False)

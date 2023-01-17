# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train_2016_v2.csv', parse_dates=['transactiondate'])



df_prop = pd.read_csv('../input/properties_2016.csv')



df_sample = pd.read_csv('../input/sample_submission.csv')



df_prop17 = pd.read_csv('../input/properties_2017.csv')



df_train17 = pd.read_csv('../input/train_2017.csv', parse_dates=['transactiondate'])

df_train.shape
df_train17.shape
pd.merge(df_train, df_train17, on='parcelid')
missing_df17 = df_prop17.isnull().sum(axis=0).reset_index()

missing_df17.columns = ['column_name', 'missing_count']

missing_df17 = missing_df17.sort_values(by='missing_count')

# missing_df17['missing_count'] /= df_prop17.shape[0]



missing_df = df_prop.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.sort_values(by='missing_count')

# missing_df['missing_count'] /= df_prop.shape[0]

# get missing value information
fig = plt.figure(figsize=(10, 8))

plt.plot(range(missing_df.shape[0]), missing_df['missing_count'])

plt.plot(range(missing_df17.shape[0]), missing_df17['missing_count'])
pd.concat([missing_df['column_name'], missing_df17['column_name']], axis=1)
tax_columns = ['structuretaxvaluedollarcnt',

       'taxvaluedollarcnt', 'landtaxvaluedollarcnt']

       # 'taxamount', 'taxdelinquencyflag', 'taxdelinquencyyear']
pd.DataFrame(df_prop[tax_columns].values / df_prop17[tax_columns].values).dropna(0, 'any')
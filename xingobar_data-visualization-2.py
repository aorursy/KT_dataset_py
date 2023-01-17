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



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train_ver2.csv',usecols=['ncodpers'])
df.head()
customer_count = df.groupby('ncodpers').agg('size').value_counts()

plt.figure(figsize=(8,6))

sns.barplot(customer_count.index,customer_count.values,color='#ffa726')

plt.xlabel('Number of occurrences of the customer')

plt.ylabel('Number of customers')
train = pd.read_csv("../input/train_ver2.csv", dtype='float16', 

                    usecols=['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 

                             'ind_cco_fin_ult1', 'ind_cder_fin_ult1',

                             'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',

                             'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',

                             'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',

                             'ind_deme_fin_ult1', 'ind_dela_fin_ult1',

                             'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',

                             'ind_hip_fin_ult1', 'ind_plan_fin_ult1',

                             'ind_pres_fin_ult1', 'ind_reca_fin_ult1',

                             'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',

                             'ind_viv_fin_ult1', 'ind_nomina_ult1',

                             'ind_nom_pens_ult1', 'ind_recibo_ult1'])
train.head()
target_sum = train.astype('float').sum(axis=0)

plt.figure(figsize=(8,6))

sns.barplot(target_sum.index,target_sum.values,color='#ffa726')

plt.xticks(rotation='vertical')

plt.xlabel('Product Name')

plt.ylabel('Number of Occurrences')
train = pd.read_csv('../input/train_ver2.csv',usecols=['fecha_dato'],parse_dates=['fecha_dato'])
train['year_month'] = train['fecha_dato'].apply(lambda x:(100*x.year) + x.month)
year_month_count  = train['year_month'].value_counts()

plt.figure(figsize=(8,6))

sns.barplot(year_month_count.index,year_month_count.values,color='#ffa726')

plt.xticks(rotation='vertical')

plt.xlabel('year and month')

plt.ylabel('Number of customers')
train = pd.read_csv('../input/train_ver2.csv',usecols=['fecha_alta'],parse_dates=['fecha_alta'])
train['year_month'] = train['fecha_alta'].apply(lambda x:(100 * x.year) + x.month)
year_month = train['year_month'].value_counts()

plt.figure(figsize=(8,6))

sns.barplot(year_month.index,year_month.values,color='#ffa726')

plt.xlabel('year and month')

plt.ylabel('Number of customers')

plt.xticks(rotation='vertical')
year_month = year_month.sort_index().reset_index()

year_month.columns = ['year_month','customers']

year_month = year_month.iloc[185:,:]

plt.figure(figsize=(12,6))

sns.barplot(year_month['year_month'].astype(int),year_month['customers'],color='#ff5722',alpha=0.75)

plt.xlabel('year and month',fontsize=12)

plt.ylabel('Number of customers',fontsize=12)

plt.xticks(rotation='vertical')
train = pd.read_csv('../input/train_ver2.csv',usecols=['age'])
train.head()
train['age'].unique()
train['age'] = train['age'].replace(' NA',np.nan)
train['age'] = train['age'].astype(float)

age_count = train['age'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(age_count.index,age_count.values.astype(int),alpha=0.75,color='#ff5722')

plt.xlabel('Age')

plt.ylabel('Number of customers')

plt.xticks(rotation='vertical')
train.isnull().sum()
print('median : ',train.age.median())

print('mean : ',train.age.mean())
train = pd.read_csv('../input/train_ver2.csv',usecols = ['renta'])
train.head()
renta_sort = pd.Series(train['renta'].unique()).sort_values()

plt.scatter(range(len(renta_sort.values)),renta_sort.values)
print('median : ',train['renta'].median())

print('mean : ',train['renta'].mean())
print('Number of null ' , train['renta'].isnull().sum())
train['renta'].fillna(train['renta'].median(),inplace=True)
quantile_series=  train['renta'].quantile(np.arange(0.99,1,0.001))

plt.figure(figsize=(8,6))

sns.barplot(quantile_series.index * 100 , quantile_series.values)

plt.xlabel('Quantile Value')

plt.ylabel('Renta Value')
renta_max = train['renta'].quantile(0.999)

train['renta'][train['renta'] > renta_max] = train['renta'].median()

sns.boxplot(train.renta.values)

plt.show()
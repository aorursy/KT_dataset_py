import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

pd.set_option('display.max_columns', None)
%matplotlib inline

train = pd.read_csv('../input/application_train.csv')
print('# of applications: {:,}'.format(train.shape[0]))
print('# of columns in train data: {}'.format(train.columns.values.size))
train.head(10)
plt.rc('font', size=14)
occ = train.groupby('OCCUPATION_TYPE').count()['TARGET']
occ = occ.sort_values(ascending=False)

sns.set_style("whitegrid")
plt.figure(figsize=(15, 5))
g = sns.barplot(x=occ.index, y=occ.values)
g.set_xticklabels(labels=occ.index, rotation=60)
g.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.title('Number of Applicantions by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.show()
plt.rc('font', size=14)
plt.figure(figsize=(20, 5))
sns.set_style("whitegrid")

plt.subplot(121)
edu = train.groupby('NAME_EDUCATION_TYPE').count()['TARGET']
edu = edu.sort_values(ascending=False)
g = sns.barplot(x=edu.index, y=edu.values)
g.set_xticklabels(labels=edu.index, rotation=20)
g.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.title('Level of Education of Applicants')
plt.xlabel('Level of Education')
plt.ylabel('Count')

plt.subplot(122)
chl = train.groupby('CNT_CHILDREN').count()['TARGET']
chl = chl.sort_values(ascending=False)
g = sns.barplot(x=chl.index, y=chl.values)
g.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.xlim(-0.3, 4)
plt.title('Number of Children Per Application')
plt.xlabel('Number of Children')
plt.ylabel('Count')
plt.show()
plt.rc('font', size=14) 
inc = train['AMT_INCOME_TOTAL']
inc = inc.sort_values(ascending=False)

plt.figure(figsize=(18, 5))
plt.subplot(121)
g = sns.distplot(inc.values[100:], kde=False)
plt.xticks(rotation=30)
g.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
g.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.title('Low Income Distribution of Applicants')
plt.xlabel('Income')
plt.ylabel('Count')

plt.subplot(122)
g = sns.distplot(inc.values[:100], kde=False)
plt.xticks(rotation=30)
g.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
g.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.title('High Income Distribution of Applicants')
plt.xlabel('Income')
plt.ylabel('Count')
plt.show()
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
pair = train[['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_REGISTRATION',
              'DAYS_LAST_PHONE_CHANGE', 'TARGET']].fillna(0)
sns.pairplot(pair)
plt.title('Correlation of Independent Vars with Target')
plt.show()
cc = pd.read_csv('../input/credit_card_balance.csv')
# pos = pd.read_csv('../input/POS_CASH_balance.csv')
# payments = pd.read_csv('../input/installments_payments.csv')
# papp = pd.read_csv('../input/previous_application.csv')
# bureau = pd.read_csv('../input/bureau.csv')
# bbalance = pd.read_csv('../input/bureau_balance.csv')

plt.rc('font', size=16)
plt.figure(figsize=(20, 7))
mean_bal = cc[['SK_ID_CURR', 'AMT_BALANCE']].groupby('SK_ID_CURR').mean()
train_bal = train[['SK_ID_CURR', 'OCCUPATION_TYPE', 'AMT_GOODS_PRICE']].join(mean_bal, on='SK_ID_CURR')
ax = sns.violinplot(x=train_bal['OCCUPATION_TYPE'], y=train_bal['AMT_BALANCE'])
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.xticks(rotation=30)
plt.xlabel('Occupation')
plt.ylabel('Average balance')
plt.title('Average Balance by Occupation')
plt.show()
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
g = sns.jointplot(x='AMT_BALANCE', y='AMT_GOODS_PRICE', data=train_bal, kind='kde', size=10, dropna=True, xlim=(0, 100000), ylim=(0, 1200000))
g.set_axis_labels("Average Balance", "Amount of Goods Price")
plt.show()
plt.rc('font', size=16)
plt.figure(figsize=(20, 7))
inc_rec = train[['SK_ID_CURR', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS']].join(mean_bal, on='SK_ID_CURR', lsuffix='train')[['NAME_INCOME_TYPE', 'AMT_BALANCE', 'NAME_FAMILY_STATUS']]

ax = sns.violinplot(x='NAME_INCOME_TYPE', y='AMT_BALANCE', data=inc_rec, palette="Set2")
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.xticks(rotation=30)
plt.xlabel('Income Type')
plt.ylabel('Average Balance Amount')
plt.title('Balance Amount by Income Type and Family Status')
plt.show()
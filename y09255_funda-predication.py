import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

sns.set(font_scale=2.5) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.

import missingno as msno

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



df_test = pd.read_csv('../input/assist-funda-comp/funda_train.csv')

submission = pd.read_csv('../input/funda-train/submission.csv')
df_test.head()
df_test.shape
df_test.info()
df_test.describe()
df_test.isnull().sum()
submission.describe()
for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))

    print(msg)
for col in submission.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (submission[col].isnull().sum() / submission[col].shape[0]))

    print(msg)
msno.matrix(df=df_test.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
msno.bar(df=df_test.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
msno.bar(df=submission.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
df_test.isnull().sum() #checking for total null values
df_test.groupby(['region', 'amount'])['amount'].count()
df_test.groupby(['type_of_business','amount'])['amount'].count()
df_test[['region', 'amount']].groupby(['region'], as_index=True).count()
df_test[['type_of_business', 'amount']].groupby(['type_of_business'], as_index=True).count()
df_test[['type_of_business', 'amount']].groupby(['type_of_business'], as_index=True).sum()
pd.crosstab(df_test.type_of_business,df_test.amount,margins=True)
pd.crosstab([df_test.region,df_test.amount],df_test.type_of_business,margins=True)
df_test.groupby(['card_company','amount'])['amount'].count()
df_test.groupby(['transacted_time','amount'])['amount'].count()
pd.crosstab(df_test.transacted_time,df_test.amount,margins=True)
df_test.groupby(['transacted_date','amount'])['amount'].count()
print('Highest amount was:',df_test['amount'].max())

print('Lowest amount was:',df_test['amount'].min())

print('Average amount was:',df_test['amount'].mean())
df_test.region.isnull().any()
df_test.type_of_business.isnull().any()
sns.heatmap(df_test.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
df_test.drop(['transacted_date','transacted_time','region','type_of_business','card_company','card_id','store_id'],axis=1,inplace=True)

sns.heatmap(df_test.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})

fig=plt.gcf()

fig.set_size_inches(18,15)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
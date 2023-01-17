import os



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
file_train = os.path.join('../input', 'train.csv')

file_test = os.path.join('../input', 'test.csv')



df_train = pd.read_csv(file_train)

df_test = pd.read_csv(file_test)
df_train.head()
df_train.describe()
#pd.plotting.scatter_matrix(df_train, alpha=0.2, figsize=(50, 50))

#plt.show()
plt.figure(figsize=(20,20))

sns.heatmap(df_train.corr(), annot=True)

plt.show()
sns.distplot(df_train['price'], hist=True, kde=True)
df_train['price'] = np.log1p(df_train['price'])
df_train['year'] = df_train['date'].apply(lambda x : str(x[:4])).astype(int)

df_test['year'] = df_test['date'].apply(lambda x : str(x[:4])).astype(int)
df_train['month'] = df_train['date'].apply(lambda x : str(x[4:6])).astype(int)

df_test['month'] = df_test['date'].apply(lambda x : str(x[4:6])).astype(int)
df_train['age'] = 2015-df_train['yr_built']

df_test['age'] = 2015-df_test['yr_built']
df_train['renovated_flg'] = None

for index, row in df_train.iterrows():

    if row['yr_renovated'] == 0:

        df_train.at[index, 'renovated_flg'] = 0

    else:

        df_train.at[index, 'renovated_flg'] = 1
df_test['renovated_flg'] = None

for index, row in df_test.iterrows():

    if row['yr_renovated'] == 0:

        df_test.at[index, 'renovated_flg'] = 0

    else:

        df_test.at[index, 'renovated_flg'] = 1
plt.figure(figsize=(20,20))

sns.heatmap(df_train.corr(), annot=True)

plt.show()
f, ax = plt.subplots(figsize=(12, 6))

data = pd.concat([df_train['price'], df_train['bedrooms']], axis=1)

sns.boxplot(x='bedrooms', y='price', data=data)
df_train.loc[df_train['bedrooms']>=10]
df_test.loc[df_test['bedrooms']>=10]
df_train = df_train.drop(df_train[(df_train['bedrooms']>=10)].index)
f, ax = plt.subplots(figsize=(12, 6))

data = pd.concat([df_train['price'], df_train['bathrooms']], axis=1)

sns.boxplot(x='bathrooms', y='price', data=data)
df_train = df_train.drop(df_train[(df_train['bathrooms']==6.25)].index)

df_train = df_train.drop(df_train[(df_train['bathrooms']==6.75)].index)

df_train = df_train.drop(df_train[(df_train['bathrooms']==7.5)].index)

df_train = df_train.drop(df_train[(df_train['bathrooms']==7.75)].index)
df_train.loc[(df_train['bathrooms']==4.5) & (df_train['price']>15)]
df_train = df_train.drop(df_train[(df_train['bathrooms']==4.5) & (df_train['price']>15)].index)
f, ax = plt.subplots(figsize=(12, 6))

data = pd.concat([df_train['price'], df_train['floors']], axis=1)

sns.boxplot(x='floors', y='price', data=data)
f, ax = plt.subplots(figsize=(12, 6))

data = pd.concat([df_train['price'], df_train['waterfront']], axis=1)

sns.boxplot(x='waterfront', y='price', data=data)
f, ax = plt.subplots(figsize=(12, 6))

data = pd.concat([df_train['price'], df_train['view']], axis=1)

sns.boxplot(x='view', y='price', data=data)
f, ax = plt.subplots(figsize=(12, 6))

data = pd.concat([df_train['price'], df_train['condition']], axis=1)

sns.boxplot(x='condition', y='price', data=data)
f, ax = plt.subplots(figsize=(12, 6))

data = pd.concat([df_train['price'], df_train['grade']], axis=1)

sns.boxplot(x='grade', y='price', data=data)
df_train.loc[df_train['grade']<=3]
df_test.loc[df_test['grade']<=3]
df_train = df_train.drop(df_train[(df_train['grade']<=3)].index)
plt.figure(figsize=(20,20))

sns.heatmap(df_train.corr().abs(), annot=True)

plt.show()
pd.plotting.scatter_matrix(df_train[['bathrooms', 'sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'price']],

                           alpha=0.2,

                          figsize=(20, 20))

plt.show()
data = pd.concat([df_train['price'], df_train['sqft_living'], df_train['bathrooms']], axis=1)

ax = sns.scatterplot(x="sqft_living", y="price", hue='bathrooms', data=data)
df_train.loc[(df_train['sqft_living']>13000) & (df_train['bathrooms']>7)]
df_train = df_train.drop(df_train.loc[(df_train['sqft_living']>13000) & (df_train['bathrooms']>7)].index)
data_src = pd.concat([df_train['price'], df_train['grade'], df_train['waterfront'], df_train['condition']], axis=1)

for condition in sorted(data_src['condition'].unique()):

    #print(condition)

    plt.title("condition: {}".format(condition))

    data = data_src[data_src['condition'] == condition]

    ax = sns.scatterplot(x="grade", y="price", hue='waterfront', data=data)

    plt.show()
data_src = pd.concat([df_train['price'], df_train['renovated_flg']], axis=1)

data = data_src.groupby(['renovated_flg']).mean()

data.plot.bar()
data_src = pd.concat([df_train['price'], df_train['view']], axis=1)

data = data_src.groupby(['view']).mean()

data.plot.bar()
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
X_train = df_train.loc[:, ~df_train.columns.isin(['id', 'price', 'date'])]

y_train = df_train['price']

X_test = df_test.loc[:, ~df_test.columns.isin(['id', 'date'])]
gbr = GradientBoostingRegressor()

scores = cross_val_score(gbr, X_train.values, y_train, cv=kfold)

rmse = np.sqrt(scores)

print("교차 검증 점수: {}".format(scores))
gbr.fit(X_train, y_train)

pred = gbr.predict(X_test)
df_submit =  pd.DataFrame(data={'id':df_test['id'],'price':np.expm1(pred)})
df_submit.to_csv('submission.csv', index=False)

print("complete!")
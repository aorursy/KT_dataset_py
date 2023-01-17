# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler

from sklearn.linear_model import Ridge

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

many_null=df.isna().sum(axis=0).sort_values(ascending=False)[:4]

print(df.isna().sum(axis=0).sort_values(ascending=False)[:4])

df.drop(many_null.index,inplace=True,axis=1)
df.describe()

df.info()
df.duplicated()
# check duplicate columns

train_enc =  pd.DataFrame(index = df.index)

for col in tqdm_notebook(df.columns):

    train_enc[col] = df[col].factorize()[0]

dup_cols = {}

for i, c1 in enumerate(tqdm_notebook(train_enc.columns)):

    for c2 in train_enc.columns[i + 1:]:

        if c2 not in dup_cols and np.all(train_enc[c1] == train_enc[c2]):

            dup_cols[c2] = c1
cat_columns=df.select_dtypes(include='object').columns

num_columns=df.select_dtypes(exclude='object').columns

plt.figure(figsize=(25,25))

corr=df.corr()

sns.heatmap(df.corr(),annot=True)
imp_columns=corr['SalePrice'][corr['SalePrice']>=0.5]

imp_columns
df[cat_columns].isna().sum(axis=0).sort_values(ascending=False)
df[num_columns].isna().sum(axis=0).sort_values(ascending=False)
similar_columns=['1stFlrSF','GarageArea']

df.drop(similar_columns,axis=1,inplace=True)
df['GarageCars']

corr=df.corr()

imp_columns=list(corr['SalePrice'][corr['SalePrice']>=0.5].index)

imp_columns=imp_columns[:len(imp_columns)-1]

print(imp_columns)
sns.distplot(df['GrLivArea'])
x=pd.get_dummies(df[cat_columns],drop_first=True)

y=np.log(df['SalePrice'])

print(cat_columns)

print(y)
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor()

clf.fit(x, y)
importances = clf.feature_importances_

z=pd.DataFrame(importances,index=x.columns,columns=['columns'])

z=z[abs(z['columns'])>0.010]

imp_forest=z.sort_values(['columns'],ascending=False)

imp_forest.plot(kind='barh',figsize=(25,15))

plt.xticks(rotation=90)
from sklearn.linear_model import Lasso,Ridge

clf=Ridge(alpha=0.06)

clf.fit(x,y)

z=pd.DataFrame(clf.coef_,index=x.columns,columns=['columns'])

z=z[abs(z['columns'])>0.3]

imp_forest=z.sort_values(['columns'],ascending=False)

imp_forest.plot(kind='barh',figsize=(25,15))

plt.xticks(rotation=90)
plt.figure(figsize=(15,15))

sns.boxplot(df['OverallQual'],df['SalePrice'])
plt.plot(df['YearRemodAdd'],df['SalePrice'],'o')

plt.figure(figsize=(15,15))

z=df[df['SalePrice']>=500000].index

df.drop(z,axis=0,inplace=True)

plt.figure(figsize=(15,15))

z=df[df['TotalBsmtSF']>=3000].index

df.drop(z,axis=0,inplace=True)

plt.plot(df['TotalBsmtSF'],df['SalePrice'],'o')
plt.figure(figsize=(15,15))

plt.plot(df['GarageCars'],df['SalePrice'],'o')
sns.distplot(np.log(df['SalePrice']))
train2 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test2 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df3 = pd.concat([train2, test2],ignore_index=True)

y=df3['SalePrice']

df3.drop('Id',inplace=True,axis=1)

num_columns=df3.select_dtypes(exclude='object').columns

df3['OverallQual']=df3['OverallQual'].transform(str)

for i in df3.columns:

    if i=='SalePrice':

        continue

    df3[i]=df3[i].fillna(df3[i].mode()[0])

print(df3['SalePrice'],df3)
num_columns
num_columns=df3.select_dtypes(exclude='object').columns

cat_columns=df3.select_dtypes(include='object').columns

#imp_columns=['YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars']

columns=[]

for i in imp_columns:

    columns.append(i);

for i in cat_columns:

    columns.append(i);

    

scaler=StandardScaler()

# SCALING THE DATA

features = df3[imp_columns]

scaler = StandardScaler().fit(features.values)

features = scaler.transform(features.values)

df3[imp_columns] = features

#print(num_columns)

for i in df3.columns:

    if i not in columns:

        if i=='SalePrice':

            continue

        df3.drop(i,inplace=True,axis=1)

num_columns=df3.select_dtypes(exclude='object').columns

print(df3.columns)

scaler.fit_transform(df3[imp_columns])

# LABELING THE DATA

df3=pd.get_dummies(df3,drop_first=True)

df3['SalePrice']

df3
processed_train = df3.loc[df3['SalePrice'].notna()]

processed_test = df3.loc[df3['SalePrice'].isna()]
processed_train['SalePrice']

#y=processed_train['SalePrice']

#y=np.log(y)
y=np.log(processed_train['SalePrice'])

processed_train.drop("SalePrice",axis=1,inplace=True)

processed_test.drop("SalePrice",axis=1,inplace=True)
clf=Ridge(alpha=0.06)

clf.fit(processed_train, y)

prediction = np.exp(clf.predict(processed_test))
submission_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission_data['SalePrice'] = prediction

submission_data.to_csv('submission.csv', index = False)
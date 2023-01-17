# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/train_2016_v2.csv") 
train.shape
train.head()
train.info()


plt.figure(figsize=(8,6))

plt.scatter(range(train.shape[0]), np.sort(train.logerror.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('logerror', fontsize=12)

plt.show()


ulogerror = train[train.logerror < train.logerror.quantile(.95)]

logerror  = ulogerror[ulogerror.logerror.quantile (0.01) < ulogerror.logerror]

plt.figure(figsize=(8,6))

plt.scatter(range(logerror.shape[0]), np.sort(logerror.logerror.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('logerror', fontsize=12)

plt.show()

plt.figure(figsize=(12,8))

sns.distplot(logerror.logerror.values, bins=50, kde=False)

plt.xlabel('logerror', fontsize=12)

plt.show()
train['transactiondate'] = pd.to_datetime(train['transactiondate'])

date = (train['transactiondate'].dt.month).value_counts()

plt.figure(figsize=(12,6))

sns.barplot(date.index,date.values,alpha=0.8,color="Blue")

plt.xlabel("Month of transcation")

plt.ylabel("Number of transcation")

plt.show()
properties = pd.read_csv("../input/properties_2016.csv") 
properties.info()
train_df = pd.merge(train, properties, on='parcelid', how='left')

train_df.head()
#check missing val

missing_df = train_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df['missing_ratio'] = missing_df['missing_count'] / train_df.shape[0]

missing_df.loc[missing_df['missing_ratio']>0]

missing_df = missing_df.sort_values(by="missing_count")

ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_df.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()

train_df = pd.merge(train, properties, on='parcelid', how='left')

train_df.head()

plt.figure(figsize=(12,12))

sns.jointplot(train_df['longitude'],train_df['latitude'])
corr = train_df.select_dtypes(include = ['float64','int64']).iloc[:,1:].corr()

plt.figure(figsize=(30, 30))

sns.heatmap(corr,vmax=1,square= True)
mean_values = train_df.mean(axis=0)

train_df = train_df.fillna(mean_values,inplace = True)

# look the correlation of each variables 

col = train_df.columns

train_float = train_df.select_dtypes(include = [np.float64])

train_float_corr = train_float.corr()

f,ax = plt.subplots(figsize = (11,9))

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

sns.heatmap(train_float_corr, cmap=cmap,square = True, linewidths = 0.5)
f,ax = plt.subplots(figsize = (11,9))

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

sns.heatmap(train_float_corr, cmap=cmap,square = True, linewidths = 0.5)


train_strong_float =train_float_corr.loc[(train_float_corr['logerror'] > 0.02) |(train_float_corr['logerror'] < -0.01)]

train_strong_float.loc[:,"logerror"]

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

x = train_df[['bathroomcnt','bedroomcnt','calculatedbathnbr','calculatedfinishedsquarefeet','fullbathcnt','heatingorsystemtypeid','structuretaxvaluedollarcnt']]

y = train_df['logerror']

X_train,X_test, y_train,y_test = train_test_split(x,y,test_size = 0.33, random_state= 42)

model = RandomForestRegressor()

model.fit(X_train,y_train)

pred = model.predict(X_test)

#evaluate 

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

print(mean_absolute_error(y_test, pred))

print(r2_score(y_test,pred))

print(mean_squared_error(y_test,pred))

plt.scatter(y_test,pred)

residual = y_test - pred

plt.scatter(residual,np.linspace(0,5,29791))
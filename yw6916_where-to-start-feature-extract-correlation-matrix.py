#standard import

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

%matplotlib inline
train_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# preview the data

train_df.head()
#train_df.info()

#print("----------------------------")

#test_df.info()
#drop unnecessary data

train_df=train_df.drop(['Id'],axis=1)

train_df.head()
train_df['SalePrice'].describe()
sns.distplot(train_df['SalePrice'])
# The intuition here is that the larger the area is, the higher the price should be.

# Therefore, we are examing this proportionality in this regards.

data = pd.concat([train_df['SalePrice'], train_df['LotArea']], axis=1)

data.plot.scatter(x='LotArea', y='SalePrice', ylim=(0, 800000))



#Well, not really see a proportionality in the chart, probably drop it.
# Similar intuition as LotArea

data = pd.concat([train_df['SalePrice'], train_df['GrLivArea']], axis=1)

data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))

#Yeah, in this case, the linear proportionality is quite obvious. Keep it.

#However, we can further clean the data by removing outliers.

train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)
data = pd.concat([train_df['SalePrice'], train_df['OverallQual']], axis=1)

#data.plot.scatter(x='OverallQual', y='SalePrice', ylim=(0, 800000))

#better version of visualization

fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#It is a quite good index. Definitely keep it.
#dropped. All the same

pass
data = pd.concat([train_df['SalePrice'], train_df['Neighborhood']], axis=1)

f, ax = plt.subplots(figsize=(26, 12))

fig = sns.boxplot(x='Neighborhood', y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)

# Different Neighborhoods have different range of price. Keep it.
#Neighborhood is proven to be less effective, ignore.



# Using Dummies to extract this feature

#neighborhood_dummies_train  = pd.get_dummies(train_df['Neighborhood'])

#neighborhood_dummies_test  = pd.get_dummies(test_df['Neighborhood'])

#train_df = train_df.join(neighborhood_dummies_train)

#test_df    = test_df.join(neighborhood_dummies_test)



#train_df.drop(['Neighborhood'], axis=1,inplace=True)

#test_df.drop(['Neighborhood'], axis=1,inplace=True)



#train_df.head()
data = pd.concat([train_df['SalePrice'], train_df['CentralAir']], axis=1)

f, ax = plt.subplots()

fig = sns.boxplot(x='CentralAir', y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#With CentralAir, the sale price is higher. Keep it for now
#Another way to handle non-numerical data.
#convert Y,N into 1,0

train_df['CentralAir'].replace(to_replace=['N', 'Y'], value=[0, 1])

train_df.head()
# This can be seen as a factor to represent the number of cars owning.

# Typically, more cars, the house tends to be more expensive

# We select GarageCars due to personal preference only

data = pd.concat([train_df['SalePrice'], train_df['GarageCars']], axis=1)

data.plot.scatter(x='GarageCars', y='SalePrice', ylim=(0, 800000))

data = pd.concat([train_df['SalePrice'], train_df['GarageArea']], axis=1)

data.plot.scatter(x='GarageArea', y='SalePrice', ylim=(0, 800000))

# Fairly representative, keeping it
# This may be a tricky one to see the correlation, since time series is involvd.

data = pd.concat([train_df['SalePrice'], train_df['YearBuilt']], axis=1)

data.plot.scatter(x='YearBuilt', y='SalePrice', ylim=(0, 800000))

#for better visualization

f, ax = plt.subplots(figsize=(26, 12))

fig = sns.boxplot(x='YearBuilt', y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#index is okay, but too many outliers, think about it later.
#This sk method can process non-value datat= like Neighborhood

from sklearn import preprocessing

f_names = ['CentralAir', 'Neighborhood']

for x in f_names:

    label = preprocessing.LabelEncoder()

    train_df[x] = label.fit_transform(train_df[x])

corrmat =train_df.corr()

f, ax = plt.subplots(figsize=(20, 9))

sns.heatmap(corrmat, vmax=0.8, square=True)
k  = 10 

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, \

                 square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
from sklearn import preprocessing

from sklearn import linear_model, svm, gaussian_process

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
cols = ['OverallQual','GrLivArea', 'GarageArea','1stFlrSF', 'FullBath', 'YearBuilt']

x = train_df[cols].values

y = train_df['SalePrice'].values

#Normalization

x_scaled = preprocessing.StandardScaler().fit_transform(x)

y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))

#Train and validation

X_train,X_vali, y_train, y_vali = train_test_split(x_scaled, y_scaled, test_size=0.3, random_state=42)
cols = ['OverallQual','GrLivArea', 'GarageArea','1stFlrSF', 'FullBath', 'YearBuilt']

X_train = train_df[cols].values

y_train = train_df['SalePrice'].values



clf_1 = RandomForestRegressor(n_estimators=400)

clf_1.fit(X_train, y_train)

y_pred = clf_1.predict(X_vali)

print(np.sum(abs(y_pred - y_vali))/len(y_pred))



clf_2 = KNeighborsRegressor(n_neighbors=7)

clf_2.fit(X_train, y_train)

y_pred = clf_2.predict(X_vali)

print(np.sum(abs(y_pred - y_vali))/len(y_pred))



clf_3 = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



clf_3.fit(X_train, y_train)

y_pred = clf_3.predict(X_vali)

print(np.sum(abs(y_pred - y_vali))/len(y_pred))
cols = ['OverallQual','GrLivArea', 'GarageArea','1stFlrSF', 'FullBath', 'YearBuilt']

test_df[cols].isnull().sum()
#Handling GarageArea Missing data

test_df['GarageArea'].describe()
test_df['GarageArea']=test_df['GarageArea'].fillna(472.768861)

test_df['GarageArea'].isnull().sum()
cols = ['OverallQual','GrLivArea', 'GarageArea','1stFlrSF', 'FullBath', 'YearBuilt']

test_x = pd.concat( [test_df[cols]] ,axis=1)



x = test_x.values



y_pred_1 = clf_1.predict(x)

y_pred_2 =clf_2.predict(x)

y_pred_3 =clf_3.predict(x)



y_pred=y_pred_1*0.5+y_pred_2*0.5
prediction = pd.DataFrame(y_pred, columns=['SalePrice'])

print(prediction)

result = pd.concat([test_df['Id'], prediction], axis=1)

result.to_csv('./Predictions.csv', index=False)
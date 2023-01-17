from matplotlib import pyplot as plt

import pandas as pd

import seaborn as sns

import numpy as np

import os

print(os.listdir("../input"))
pd.set_option('display.max_columns',100)
train = pd.read_csv(r'../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape
train.SalePrice.skew()
train.describe()
var = 'OverallQual'

data = pd.concat([train['SalePrice'],train[var]],axis=1)

plt.figure(figsize=(8,6),dpi=80)

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'GrLivArea'

data = pd.concat([train[var],train['SalePrice']],axis=1)

data.plot.scatter(x=var,y='SalePrice')
corrmat = train.corr()

plt.figure(figsize=(12,12),dpi=80)

sns.heatmap(corrmat,square=True,cmap='YlGnBu')
index =corrmat.drop(['Id'],axis=1).nlargest(10,'SalePrice').index

c10 = train[index]

# c10

plt.figure(figsize=(12,12))

sns.heatmap(c10.corr(),cbar=True,annot=True,square=True,cmap='YlGnBu')
c10.describe()
O = 'OverallQual'

P = 'SalePrice'

data = pd.concat([train[O],train[P]],axis =1)

sns.boxplot(x=O,y=P,data=data)
G = 'GrLivArea'

train.plot.scatter(x=G,y=P)
train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index,inplace=True)
train.plot.scatter(x=G,y=P)
sns.distplot(train[P])
train[P] = np.log1p(train[P])
sns.distplot(train[P])
y_train = train[P]

all_data = pd.concat((train.drop([P],axis=1), test),axis=0,ignore_index=True,sort=False).drop(['Id'],axis=1)
all_data
all_na = all_data.isnull().sum().sort_values(ascending=False)
all_na[all_na!=0]
all_data[all_na.index]
for i in all_na.index[:5]:

    all_data[i]=all_data[i].fillna('None')
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
#车库的事

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
all_data = all_data.drop(['Utilities'], axis=1)

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
all_data['MSSubClass'].value_counts()
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



all_data['OverallCond'] = all_data['OverallCond'].astype(str)



all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data_dumies = pd.get_dummies(all_data)
all_data_dumies
y_train = train[P]

train.shape
test.shape
all_data.shape
x_train = all_data_dumies[:train.shape[0]]

x_test = all_data_dumies[train.shape[0]:]
x_train
x_test
from xgboost.sklearn import XGBRegressor

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingRegressor
rr = Ridge(alpha=10)

rr.fit(x_train, y_train)

np.sqrt(-cross_val_score(rr, x_train, y_train, cv=5, scoring="neg_mean_squared_error")).mean()
y_pred = rr.predict(x_train)

# 误差

resid = (y_train-y_pred)

# 平均误差

mean_resid = resid.mean()

# 误差标准差

std_resid = resid.std()

# Z

z = np.array((resid-mean_resid)/std_resid)

# 异常值

outliers1 = np.where(abs(z) > abs(z).std() * 3)[0]

outliers1





plt.figure(figsize=(6, 6))

plt.scatter(y_train, y_pred)

plt.scatter(y_train.iloc[outliers1], y_pred[outliers1])

plt.plot(range(10, 15), range(10, 15), color="red")
x_train = x_train.reset_index(drop=True).drop(outliers1)

y_train = y_train.reset_index(drop=True).drop(outliers1)
xgb = XGBRegressor(n_estimators=600,n_jobs=-1,max_depth=5,colsample_bytree=0.5,subsample=0.8)

xgb.fit(x_train, y_train)

np.sqrt(-cross_val_score(xgb, x_train, y_train, cv=5, scoring="neg_mean_squared_error")).mean()
gbdt = GradientBoostingRegressor(max_depth=5, n_estimators=600,max_features=0.5)

gbdt.fit(x_train,y_train)

np.sqrt(-cross_val_score(gbdt,x_train,y_train,cv=5, scoring="neg_mean_squared_error")).mean()
rr = Ridge()

rr.fit(x_train, y_train)

np.sqrt(-cross_val_score(rr, x_train, y_train, cv=5, scoring="neg_mean_squared_error")).mean()
lsr = Lasso(alpha=0.00047)

lsr.fit(x_train, y_train)

np.sqrt(-cross_val_score(lsr, x_train, y_train, cv=5, scoring="neg_mean_squared_error")).mean()
# 模型组合

train_pred = 0.2*xgb.predict(x_train)+0.5*rr.predict(x_train)+0.3*lsr.predict(x_train)
plt.figure(figsize=(12,12))

plt.scatter(y_train,train_pred)

tick = [i/10 for i in range(109,137)]

plt.plot(tick,tick,c='r')

plt.xticks([i/10 for i in range(109,137)])

plt.xticks(tick)

plt.yticks(tick)

plt.grid()
pd.DataFrame(train_pred).quantile(0.0046

                                 )
q1 = pd.DataFrame(train_pred).quantile(0.0046)

pre_df = pd.DataFrame(train_pred)

pre_df["SalePrice"] = train_pred

pre_df = pre_df[["SalePrice"]]

pre_df.loc[pre_df.SalePrice <= q1[0], "SalePrice"] = pre_df.loc[pre_df.SalePrice <= q1[0], "SalePrice"] *0.9909

train_pred = np.array(pre_df.SalePrice)

plt.figure(figsize=(12,12))

plt.scatter(y_train,train_pred)

tick = [i/10 for i in range(109,137)]

plt.plot(tick,tick,c='r')

plt.xticks(tick)

plt.yticks(tick)

plt.grid()
submission=pd.read_csv('../input/sample_submission.csv')

test_predict =+ 0.2 * xgb.predict(x_test) + 0.3 * lsr.predict(x_test) + 0.5 * rr.predict(x_test)

q1 = pd.DataFrame(test_predict).quantile(0.0042)

pre_df = pd.DataFrame(test_predict)

pre_df["SalePrice"] = test_predict

pre_df = pre_df[["SalePrice"]]

pre_df.loc[pre_df.SalePrice <= q1[0], "SalePrice"] = pre_df.loc[pre_df.SalePrice <= q1[0], "SalePrice"] *0.98

test_predict = np.array(pre_df.SalePrice)

submission["SalePrice"] = np.exp(test_predict)-1

submission.to_csv("submission.csv", index=False)
submission
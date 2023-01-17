# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#bring in the six packs
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

datasets = [df_train, df_test]
df_train.head()
numericFields = ['SalePrice', 'LotArea', 'LotFrontage', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',  '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2',  'GarageArea', 'PoolArea']
for field in numericFields:
    plt.figure()
    sns.distplot(df_train.loc[~df_train[field].isnull()][field])
numericFields = ['SalePrice', 'LotArea', 'LotFrontage', 'MasVnrArea', 'BsmtUnfSF', '1stFlrSF']
for field in numericFields:
    plt.figure()
    sns.distplot(np.log1p(df_train.loc[~df_train[field].isnull()][field]))
print(df_train['SalePrice'].skew()) # apply log to convert to normal dist.
df_train['SalePrice'].describe()
df_train['SalePrice'].isnull().sum() # no null values.
df_train.BedroomAbvGr.describe()
data = pd.concat([df_train['SalePrice'], df_train['BedroomAbvGr']], axis=1)
data.plot.scatter(x='BedroomAbvGr', y='SalePrice')
data = pd.concat([df_train['SalePrice'], df_train['LotArea']], axis=1)
data.plot.scatter(x='LotArea', y='SalePrice')
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice')
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
cols = corrmat.nlargest(10, 'SalePrice').index
corrmat_small = df_train[cols].corr()
hm = sns.heatmap(corrmat_small)
proportion_null = pd.DataFrame(df_train.isnull().sum() / df_train.isnull().count(), columns=['Prop Null']).sort_values(by="Prop Null", ascending=False).head(10)
proportion_null
df_train.LotFrontage.describe()
pool_stats = pd.concat([df_train.PoolArea, df_train.PoolQC], axis=1)
zero_pool = pool_stats[pool_stats.PoolArea == 0]
zero_pool.PoolQC.isnull().describe()

for dataset in datasets:
    dataset.HasPool = dataset.PoolArea != 0
print(set(df_train.PoolQC))
quality_map = {np.nan: 0, 'Fa': 1, 'Gd': 2, 'Ex': 3}
for dataset in datasets:
    dataset.PoolQC = dataset.PoolQC.map(quality_map)
print(set(df_train.MiscFeature))
feature_map = {np.nan: 0, 'TenC': 1, 'Gar2': 2, 'Othr': 3, 'Shed': 4}
for dataset in datasets:
    dataset.MiscFeature = dataset.MiscFeature.map(feature_map)
set(df_train.Alley)
alley_map = {np.nan: 0, 'Pave': 1, 'Grvl': 2}
for dataset in datasets:
    dataset.Alley = dataset.Alley.map(alley_map)
set(df_train.Fence)
fence_map = {np.nan: 0, 'MnWw': 1, 'GdPrv': 2, 'GdWo': 3, 'MnPrv': 4}
for dataset in datasets:
    dataset.Fence = dataset.Fence.map(fence_map)
df_train.groupby('FireplaceQu').count().Id.plot(kind='bar')
fireplace_qu_mapping = {np.nan: 0, 'Po': 1, 'Fa':2, 'TA': 3, 'Gd': 4, 'Ex': 5}
for dataset in datasets:
    dataset.FireplaceQu = dataset.FireplaceQu.map(fireplace_qu_mapping)

data = pd.concat([df_train.LotFrontage, df_train.SalePrice], axis=1)
data.plot.scatter(x="LotFrontage", y="SalePrice")
print(df_train.LotFrontage.describe())
cols = corrmat.nlargest(10, 'LotFrontage').index
corrmat_small = df_train[cols].corr()
hm = sns.heatmap(corrmat_small)
# for each property, if the lot frontage is null, then find the property with the closest lot area that is not null, and fill it with this value. 
def remove_lot_frontage_nulls(dataset):
    null_frontages = dataset.loc[dataset.LotFrontage.isnull()]
    non_null_frontages = dataset.loc[~dataset.LotFrontage.isnull()]
    new_frontages_rows = []
    for first_floor_sf in null_frontages['1stFlrSF']:
        df_sort = non_null_frontages.iloc[(non_null_frontages['1stFlrSF'] - first_floor_sf).abs().argsort()[:1]]
        new_frontages_rows.append(df_sort)
    new_frontages_rows = pd.concat(new_frontages_rows)
    new_frontages_rows.index = null_frontages.index
    lotFrontageNoNa = dataset.LotFrontage.dropna()
    print(lotFrontageNoNa.head(10))
    print(new_frontages_rows.LotFrontage.head(10))
    dataset.LotFrontage = pd.concat([lotFrontageNoNa, new_frontages_rows.LotFrontage])
    return dataset

df_train = remove_lot_frontage_nulls(df_train)
df_test = remove_lot_frontage_nulls(df_test)
datasets = [df_train, df_test]
df_train.LotFrontage.describe()
proportion_null = pd.DataFrame(df_train.isnull().sum() / df_train.isnull().count(), columns=['Prop Null']).sort_values(by="Prop Null", ascending=False).head(20)
proportion_null
set(df_train.GarageQual)
garage_qual_mapping = {np.nan: 0, 'Po': 1, 'Fa':2, 'TA': 3, 'Gd': 4, 'Ex': 5}
garage_fin_mapping = {np.nan: 0, 'RFn': 1, 'Unf': 2, 'Fin': 3}
garage_type_mapping = {np.nan: 0, 'CarPort': 1, 'Attchd': 2, 'Detchd': 3, '2Types': 4, 'Basment': 5, 'BuiltIn': 6}

for dataset in datasets:
    dataset.GarageQual = dataset.GarageQual.map(garage_qual_mapping)
    dataset.GarageCond = dataset.GarageCond.map(garage_qual_mapping)
    dataset.GarageFinish = dataset.GarageFinish.map(garage_fin_mapping)
    dataset.GarageType = dataset.GarageType.map(garage_type_mapping)
print (set(df_train.BsmtFinType2))
bsmt_qual_mapping = {np.nan: 0, 'Po': 1, 'Fa':2, 'TA': 3, 'Gd': 4, 'Ex': 5}
bsmt_fin_types = {np.nan: 0, 'Rec': 1, 'Unf': 2, 'ALQ': 3, 'LwQ': 4, 'GLQ': 5, 'BLQ': 6}
bsmt_exposure = {np.nan:0, 'Mn': 1,'Gd': 2, 'No': 3, 'Av': 4}
for dataset in datasets:
    dataset.BsmtQual = dataset.BsmtQual.map(bsmt_qual_mapping)
    dataset.BsmtCond = dataset.BsmtCond.map(bsmt_qual_mapping)
    dataset.BsmtFinType1 = dataset.BsmtFinType1.map(bsmt_fin_types)
    dataset.BsmtFinType2 = dataset.BsmtFinType2.map(bsmt_fin_types)
    dataset.BsmtExposure = dataset.BsmtExposure.map(bsmt_exposure)
sns.distplot(df_train.loc[~df_train.GarageYrBlt.isnull()].GarageYrBlt)
sns.distplot(df_train.loc[~df_train.YearBuilt.isnull()].YearBuilt)
data = pd.concat([df_train.GarageYrBlt, df_train.YearBuilt], axis=1)
corrmat = data.corr()
corrmat
for dataset in datasets:
    dataset.GarageYrBlt.fillna(dataset.YearBuilt, inplace=True)
for dataset in datasets:
    dataset.MasVnrType.fillna(dataset.groupby('MasVnrType').count().Id.idxmax(), inplace=True)
sns.distplot(df_train.loc[~df_train.MasVnrArea.isnull()].MasVnrArea)
for dataset in datasets:
    dataset.MasVnrArea.fillna(dataset.MasVnrArea.median(), inplace=True)
elec_types = {np.nan: 0, 'FuseA': 1, 'FuseF': 2, 'Mix': 3, 'SBrkr': 4, 'FuseP': 5}
set(df_train.Electrical)
for dataset in datasets:
    dataset.Electrical = dataset.Electrical.map(elec_types)
# binnable = ['LotArea','TotalBsmtSF', 'BsmtFinSF1', 'GrLivArea', '1stFlrSF']
# for dataset in datasets:
#     for field in binnable:
#         dataset[field] = pd.cut(dataset[field], 5, labels=np.arange(5))
y = df_train['SalePrice'] 
X = df_train.drop('SalePrice', axis=1)
X = pd.get_dummies(X)
Xt = pd.get_dummies(df_test)
X, Xt = X.align(Xt, join='inner', axis=1)
X.fillna(X.mean(), inplace=True)
Xt.fillna(Xt.mean(), inplace=True)
proportion_null = pd.DataFrame(X.isnull().sum() / X.isnull().count(), columns=['Prop Null']).sort_values(by="Prop Null", ascending=False).head(20)
proportion_null
# Partition the dataset in train + validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv(model, X, y):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring = scorer, cv = 10))
    return(rmse)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
# Look at predictions on training and validation set
print("RMSE on Training set :", rmse_cv(model, X_train, y_train).mean())
print("RMSE on Test set :", rmse_cv(model, X_test, y_test).mean())
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Plot residuals
plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()
X.describe().std().sort_values(ascending=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

y_train = np.log(y_train)
y_test = np.log(y_test) 

model = RandomForestRegressor()
model.fit(X_train, y_train)
# Look at predictions on training and validation set
y_train = np.exp(y_train)
y_test = np.exp(y_test)

print("RMSE on Training set :", rmse_cv(model, X_train, y_train).mean())
print("RMSE on Test set :", rmse_cv(model, X_test, y_test).mean())
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train_pred = np.exp(y_train_pred)
y_test_pred = np.exp(y_test_pred)

# Plot residuals
plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

y_train = np.log(y_train)
y_test = np.log(y_test) 

# Let's try XGboost algorithm to see if we can get better results
model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

model.fit(X_train, y_train)
# Look at predictions on training and validation set
y_train = np.exp(y_train)
y_test = np.exp(y_test)

print("RMSE on Training set :", rmse_cv(model, X_train, y_train).mean())
print("RMSE on Test set :", rmse_cv(model, X_test, y_test).mean())
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train_pred = np.exp(y_train_pred)
y_test_pred = np.exp(y_test_pred)

# Plot residuals
plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()
y_t = np.log(y) 

# rf = RandomForestRegressor(n_estimators=100)
# rf.fit(X, y)

model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
model.fit(X, y_t)

preds = model.predict(Xt)
preds = np.exp(preds)
preds
submission = pd.DataFrame({"Id": Xt["Id"],"SalePrice": preds})
submission.loc[submission['SalePrice'] <= 0, 'SalePrice'] = 0
fileName = "submissioning.csv"
submission.to_csv(fileName, index=False)
!cat submissioning.csv

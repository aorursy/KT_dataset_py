# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import shap
from numpy import sort
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
#from sklearn.impute import SimpleImputer """!!! SimpleImpute doesn't work anymore"""
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.kernel_ridge import KernelRidge
from math import sqrt
from scipy import stats
from scipy.stats import norm, skew
import seaborn as sns
import eli5
from eli5.sklearn import PermutationImportance
import tensorflow as tf
from tensorflow import keras
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)



def check_skewness(col):
    sns.distplot(train_data_raw[col] , fit=norm);
    fig = pyplot.figure()
    res = stats.probplot(train_data_raw[col], plot=pyplot)
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(train_data_raw[col])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    

import os
print(os.listdir("../input"))


"""Import test and train"""

train_data_raw = pd.read_csv("../input/train.csv")
test_data_raw = pd.read_csv("../input/test.csv")

#train_data_raw.fillna(-999, inplace = True)
#test_data_raw.fillna(-999, inplace = True)

y = train_data_raw.SalePrice

#print (train_data_raw.info())
#print (test_data_raw.info())
print (y.head(5))
# Any results you write to the current directory are saved as output.
full_set = pd.concat([train_data_raw, test_data_raw]).reset_index(drop=True)
print (full_set.shape)


nan_features = full_set.columns[full_set.isna().any()].tolist()

print (nan_features)
#full_set[nan_features]
full_set[(full_set['TotalBsmtSF'] > 0) & (full_set['BsmtCond'].isnull() | full_set['BsmtExposure'].isnull() |  full_set['BsmtFinSF1'].isnull() | full_set['BsmtFinSF2'].isnull() |
                                          full_set['BsmtFinType1'].isnull() | full_set['BsmtFinType2'].isnull() |
                                          full_set['BsmtFullBath'].isnull()| full_set['BsmtHalfBath'].isnull() | 
                                          full_set['BsmtQual'].isnull() | full_set['BsmtUnfSF'].isnull()                                         
                                         )]
full_set.loc[2040, 'BsmtCond'] = 'Gd'
full_set.loc[2185, 'BsmtCond'] = 'TA'

full_set.loc[332, 'BsmtFinType2'] = 'Rec'

full_set.loc[2217, 'BsmtQual'] = 'Fa'
full_set.loc[2218, 'BsmtQual'] = 'TA'

full_set.loc[948, 'BsmtExposure'] = 'No'
full_set.loc[1487, 'BsmtExposure'] = 'No'
full_set.loc[2348, 'BsmtExposure'] = 'No'

list_to_remove = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF']

nan_features = list(set(nan_features).difference(set(list_to_remove)))

print (nan_features)
full_set[full_set['GarageQual'].isnull() & (full_set['GarageType'] == 'Detchd')]
full_set.loc[2576, 'GarageArea'] = full_set['GarageArea'].median()

full_set.loc[2576, 'GarageCars'] = full_set['GarageCars'].median()

full_set.loc[2126, 'GarageCond'] = full_set['GarageCond'].mode()[0]
full_set.loc[2576, 'GarageCond'] = full_set['GarageCond'].mode()[0]

full_set.loc[2126, 'GarageFinish'] = full_set['GarageFinish'].mode()[0]
full_set.loc[2576, 'GarageFinish'] = full_set['GarageFinish'].mode()[0]

full_set.loc[2126, 'GarageQual'] = full_set['GarageQual'].mode()[0]
full_set.loc[2576, 'GarageQual'] = full_set['GarageQual'].mode()[0]

full_set.loc[2126, 'GarageYrBlt'] = full_set['GarageYrBlt'].median()
full_set.loc[2576, 'GarageYrBlt'] = full_set['GarageYrBlt'].median()

list_to_remove = ['GarageQual', 'GarageYrBlt', 'GarageFinish', 'GarageArea','GarageCond','GarageCars', 'GarageType']

nan_features = list(set(nan_features).difference(set(list_to_remove)))

print (nan_features)
"""Checks for nulls"""
full_set[(full_set['Fireplaces'] > 0) & (full_set['FireplaceQu'].isnull())] #none

full_set[(full_set['TotalBsmtSF'].isnull())] #ok
full_set.loc[2120, 'TotalBsmtSF'] = 0


full_set[full_set['SaleType'].isnull()]
full_set.loc[2489, 'SaleType'] = full_set['SaleType'].mode()[0]
full_set[full_set['KitchenQual'].isnull()]
full_set.loc[1555, 'KitchenQual'] = 'TA'
full_set[full_set['Functional'].isnull()]
full_set['Functional'] = full_set['Functional'].fillna('Typ')
full_set['Electrical'] = full_set['Electrical'].fillna('SBrkr')
full_set['Exterior1st'] = full_set['Exterior1st'].fillna(full_set['Exterior1st'].mode()[0])
full_set['Exterior2nd'] = full_set['Exterior2nd'].fillna(full_set['Exterior2nd'].mode()[0])
nan_features = full_set.columns[full_set.isna().any()].tolist()

print (nan_features)
#check BsmtFinSF1, BsmtFinSF2, BsmtunfSF, MasVnrArea, MiscFeature, PoolQC with PoolArea, 
#super check MasVnrType, MSZoning, Utilities
#MasVnrArea ok
full_set[full_set['MasVnrType'].isnull()]
full_set.loc[2610, 'MasVnrType'] = full_set['MasVnrType'].mode()[0]
full_set[full_set['MSZoning'].isnull()]
full_set.loc[1915, 'MSZoning'] = "RM"
full_set.loc[2216, 'MSZoning'] = "RL"
full_set.loc[2250, 'MSZoning'] = "RM"
full_set.loc[2904, 'MSZoning'] = "RL"

full_set[full_set['MSZoning'].isnull()]
full_set[full_set['Utilities'].isnull()]
full_set.loc[1915, "Utilities"] = full_set["Utilities"].mode()[0]
full_set.loc[1945, "Utilities"] = full_set["Utilities"].mode()[0]

full_set[full_set['Utilities'].isnull()]
full_set[(full_set['PoolArea'] > 0) & (full_set['PoolQC'].isnull())]
full_set.loc[2420, "PoolQC"] = full_set["PoolQC"].mode()[0]
full_set.loc[2503, "PoolQC"] = full_set["PoolQC"].mode()[0]
full_set.loc[2599, "PoolQC"] = full_set["PoolQC"].mode()[0]

full_set[(full_set['PoolArea'] > 0) & (full_set['PoolQC'].isnull())]
full_set[full_set['SalePrice'].isnull()]
#full_set.fillna(-999, inplace = True)



print(full_set.info())
objects = []
for i in full_set.columns:
    if full_set[i].dtype == object:
        objects.append(i)

full_set.update(full_set[objects].fillna('None'))
nulls = np.sum(full_set.isnull())
nullcols = nulls.loc[(nulls != 0)]
dtypes = full_set.dtypes
dtypes2 = dtypes.loc[(nulls != 0)]
info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)
print(info)
print("There are", len(nullcols), "columns with missing values")
full_set['LotFrontage'] = full_set.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
nulls = np.sum(full_set.isnull())
nullcols = nulls.loc[(nulls != 0)]
dtypes = full_set.dtypes
dtypes2 = dtypes.loc[(nulls != 0)]
info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)
print(info)
print("There are", len(nullcols), "columns with missing values")
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in full_set.columns:
    if full_set[i].dtype in numeric_dtypes: 
        numerics.append(i)
        
full_set.update(full_set[numerics].fillna(0))
nulls = np.sum(full_set.isnull())
nullcols = nulls.loc[(nulls != 0)]
dtypes = full_set.dtypes
dtypes2 = dtypes.loc[(nulls != 0)]
info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)
print(info)
print("There are", len(nullcols), "columns with missing values")
full_set['TotalSF'] = full_set['TotalBsmtSF'] + full_set['1stFlrSF'] + full_set['2ndFlrSF']
numeric_feats = full_set.dtypes[full_set.dtypes != "object"].index
skewed_feats = full_set[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

#.sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.drop('SalePrice',axis = 0)
skewness.head(25)


skewness.drop(['SalePrice','Id'], inplace = True)    
print (skewness.index)
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    full_set[feat] = boxcox1p(full_set[feat], lam)
full_set = pd.get_dummies(full_set)
train_data_raw = full_set.iloc[:1460,:]
test_data_raw = full_set.iloc[1460:,:]

print (train_data_raw.shape, test_data_raw.shape)
"""Analyze data"""

corrmat = train_data_raw.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
pyplot.figure(figsize=(10,10))
g = sns.heatmap(train_data_raw[top_corr_features].corr(),annot=True, vmax=0.9, square=True)


"""

fig, ax = pyplot.subplots()
ax.scatter(x = train_data_raw['GrLivArea'], y = y)
pyplot.ylabel('SalePrice', fontsize=13)
pyplot.xlabel('GrLivArea', fontsize=13)
pyplot.show()
"""
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_data_raw[cols], height = 3)
pyplot.show();

check_skewness('SalePrice')

print (top_corr_features)
train_data_raw["SalePrice"] = np.log1p(train_data_raw["SalePrice"])

check_skewness('SalePrice')
"""Data preparation"""

train_data_raw = train_data_raw.drop(train_data_raw[(train_data_raw['GrLivArea']>4000) & (train_data_raw['SalePrice']<300000)].index)
y = train_data_raw.SalePrice
"""Uncomment all for XGBoost"""

#Droping unnecesarry features

final_train = train_data_raw.drop(columns = ['Id','SalePrice'])
test_data = test_data_raw.drop(columns = ['Id','SalePrice'])

#print (train_data.info())


"""
temp = train_data
final_train = pd.DataFrame(train_imputer.fit_transform(train_data))
final_train.columns = temp.columns

temp2 = test_data
test_data = pd.DataFrame(train_imputer.fit_transform(test_data))
test_data = temp2.columns
#print (final_train.columns.duplicated())
"""

#print (final_train.info(), test_data.info())


"""Uncomment all for XGBoost"""

final_train.describe()

#feature_names = [i for i in final_train.columns if final_train[i].dtype in [np.float64]]
feature_names = list(final_train)

#print (feature_names)
"""Uncomment all for XGBoost"""


train_X, val_X, train_y, val_y = train_test_split(final_train, y, random_state = 1)

X = final_train
train_X = train_X.as_matrix()
train_y = train_y.as_matrix()
val_X = val_X.as_matrix()
val_y = val_y.as_matrix()

test_model = XGBRegressor().fit(train_X, train_y)
#val_X = val_X[train_X.columns]
perm = PermutationImportance(test_model, random_state = 1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = list(X))

"""Uncomment all for XGBoost"""



#Shap feature selection
from sklearn.feature_selection import SelectFromModel
#shap.initjs()


sel = SelectFromModel(perm, threshold=0.00001, prefit = True)
final_train_trans = sel.transform(final_train)

feature_idx = sel.get_support()
feature_name = final_train.columns[feature_idx]
kek_feat = list(top_corr_features)
kek_feat.remove('SalePrice')
print (kek_feat)
print (list(feature_name))
features_to_keep = list(feature_name)
train_data_for_reqs = final_train
#[features_to_keep]
"""Uncomment all for XGBoost"""



"""Catboost"""

"""
from catboost import Pool, CatBoostRegressor, cv
import hyperopt
cat_train_data = train_data_raw.drop(columns = ['Id','SalePrice'])
cat_test_data = test_data_raw.drop(columns = ['Id','SalePrice'])
#cat_train_data = cat_train_data[features_keeping_cat]
"""
"""
#Catboost

numeric_feats = cat_train_data.dtypes[cat_train_data.dtypes != "object"].index
skewed_feats = cat_train_data[numeric_feats].apply(lambda x: skew(x.dropna()))
#.sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

"""
"""
#Catboost

skewness = skewness[abs(skewness) > 0.75]


print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    cat_train_data[feat] = boxcox1p(cat_train_data[feat], lam)
    cat_test_data[feat] = boxcox1p(cat_test_data[feat], lam)
    
"""
"""
#Catboost
print(cat_train_data.info())
print(cat_test_data.info())
"""
"""
#Catboost
cat_feat = np.where(cat_train_data.dtypes == object)[0]

train_X, val_X, train_y, val_y = train_test_split(cat_train_data, y)
params = {'iterations':3000,
          'depth': 3, #
          'use_best_model': True,
          'loss_function': 'RMSE',
          'eval_metric': 'RMSE',
          'l2_leaf_reg': 2,#2 (1311)
          'rsm': 0.1 #0.1 (1409)
          }

#0.15
poolz = Pool(cat_train_data, y, cat_features = cat_feat)

catmodel = CatBoostRegressor(**params)

catmodel.fit(train_X, train_y, cat_features = cat_feat, eval_set = (val_X, val_y))


"""
#Do no uncomment
"""
scores_cat = cv(params = params, 
                pool = poolz, 
                fold_count=5)
"""
#print(scores_cat.mean())


"""for i in range(0,len(cat_scores_test)):
    cat_scores_test[i] = sqrt(cat_scores_test[i] * -1)
    print (cat_scores_test[i])
print (cat_scores_test.mean())"""
"""Modelling with XGBoost"""

#low_skew_features.remove('HouseStyle_2.5Fin')
#print (train_data_for_reqs.info())
#print (low_skew_features)
#train_data_for_reqs = train_data_for_reqs[low_skew_features]
#train_data_for_reqs.drop(columns = ['HouseStyle_2.5Fin'])
#[features_to_keep]
#print (train_data_for_reqs.info())


test_pipeline = make_pipeline(Imputer(), 
                XGBRegressor(n_estimators = 4000, 
                        max_depth = 3, #3
                        learning_rate = 0.05,#0.05
                        min_child_weight = 1, #3
                        early_stopping_rounds = 20,
                        reg_alpha = 1,#1
                        reg_lambda = 1,#1
                        subsample = 0.7,
                        colsample_bytree = 0.1,
                        eval_metric = 'rmse', 
                        verbose = True))
#0112
scores_test = cross_val_score(test_pipeline, train_data_for_reqs, y, scoring = 'neg_mean_squared_error',cv = 5)
for i in range(0,len(scores_test)):
    scores_test[i] = sqrt(scores_test[i] * -1)
    print (scores_test[i])
print (scores_test.mean())


test_pipeline.fit(train_data_for_reqs, y)
#predicted = cross_val_predict(test_pipeline, train_data_for_reqs, y, cv=5)
#print (metrics.mean_absolute_error(y, predicted))

"""Kernel Ridge"""

test_pipeline_ridge = make_pipeline(Imputer(), 
                KernelRidge(alpha = 0.8,
                            kernel = 'polynomial',
                            degree = 2,
                            coef0 = 3
                            ))
#0112
scores_test_ridge = cross_val_score(test_pipeline_ridge, train_data_for_reqs, y, scoring = 'neg_mean_squared_error',cv = 5)
for i in range(0,len(scores_test_ridge)):
    scores_test_ridge[i] = sqrt(scores_test_ridge[i] * -1)
    print (scores_test_ridge[i])
print (scores_test_ridge.mean())


test_pipeline_ridge.fit(train_data_for_reqs, y)
#predicted = cross_val_predict(test_pipeline, train_data_for_reqs, y, cv=5)
#print (metrics.mean_absolute_error(y, predicted))

"""

#Catboost

cat_train_data.info()
cat_test_data.info()

"""
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(256, activation = tf.nn.relu,
                          input_shape = (train_data_for_reqs.shape[1],)),
        keras.layers.Dense(256, activation = tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile (loss = 'mse', optimizer = optimizer, metrics = ['mse'])
    return model

model = build_model()
model.summary()
train_X, val_X, train_y, val_y = train_test_split(train_data_for_reqs, y, random_state = 5)
trainer = model.fit(train_X, train_y, epochs = 800, validation_data = (val_X, val_y))
def plot_history(trainer, x, y):
  pyplot.figure()
  pyplot.xlabel('Epoch')
  pyplot.ylabel('Mean Abs Error')
  pyplot.plot(trainer.epoch, np.array(trainer.history['mean_squared_error']),
           label='Train Loss')
  pyplot.plot(trainer.epoch, np.array(trainer.history['val_mean_squared_error']),
           label = 'Val loss')
  pyplot.legend()
  pyplot.ylim([x, y])

plot_history(trainer, 0, 2)

[loss, mse] = model.evaluate(train_data_for_reqs, y)
print ('rootz', np.sqrt(mse))
train_X, val_X, train_y, val_y = train_test_split(train_data_for_reqs, y, random_state = 42)
trainer = model.fit(train_X, train_y, epochs = 1800, validation_data = (val_X, val_y))
plot_history(trainer, 0, 0.05)
[loss, mse] = model.evaluate(train_data_for_reqs, y)
print ('rootz', np.sqrt(mse))

final_test = test_data
#[features_to_keep]
predictedTf = np.expm1(model.predict(final_test).flatten())
predicted = np.expm1(test_pipeline.predict(final_test))
predictedMN = np.mean(np.array([predictedTf, predicted]), axis = 0)
print (predicted[0:5])
print (predictedTf[0:5])
print (predictedMN)
output = pd.DataFrame({'Id': test_data_raw.Id,
                       'SalePrice': predictedMN})


#catmodel.fit(train_X, train_y, cat_features = cat_feat, eval_set = (val_X, val_y))
"""
#Uncomment for catboost
cat_predictions = np.expm1(catmodel.predict(cat_test_data))
print (cat_predictions[0:5])

output = pd.DataFrame({'Id': test_data_raw.Id,
                       'SalePrice': cat_predictions})
"""


output.to_csv('submission.csv', index=False)

print ("job done")



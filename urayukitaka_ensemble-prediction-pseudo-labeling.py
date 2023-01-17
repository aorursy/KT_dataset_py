# Basic library

import numpy as np 

import pandas as pd

import time

import warnings

warnings.simplefilter("ignore")



# Data preprocessing

import category_encoders as ce



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# models of regressions

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

import lightgbm as lgb

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet



# stacking regressor

from sklearn.ensemble import StackingRegressor



# BayesSearchCV

from skopt import BayesSearchCV



# Cross validation

from sklearn.model_selection import cross_val_score



# R2 score

from sklearn.metrics import r2_score



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Data loading

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sample = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
# drop index columns

train.drop("Id", axis=1, inplace=True)

test.drop("Id", axis=1, inplace=True)
# Separate target values from train data

target = train["SalePrice"]

train = train.drop("SalePrice", axis=1)
# train data null counts

train.isnull().sum().sum()
test.isnull().sum().sum()
# MSZoning of test data

test["MSZoning"].fillna("RL", inplace=True)



# LotFrontage, fill 0

train["LotFrontage"].fillna(0, inplace=True)

test["LotFrontage"].fillna(0, inplace=True)



# Alley, Fill na by "No_alley"

train["Alley"].fillna("No_alley", inplace=True)

test["Alley"].fillna("No_alley", inplace=True)



# Utilities of test data, drop columns from train and test

train.drop("Utilities", axis=1, inplace=True)

test.drop("Utilities", axis=1, inplace=True)



# Exterior1st and Extrior2nd, of test data, fill 1st mode type

test["Exterior1st"].fillna("VinylSd", inplace=True)

test["Exterior2nd"].fillna("VinylSd", inplace=True)



# MasVnrType, fill by "None"

train["MasVnrType"].fillna("None", inplace=True)

test["MasVnrType"].fillna("None", inplace=True)



# MasVnrArea, fill by 0

train["MasVnrArea"].fillna(0, inplace=True)

test["MasVnrArea"].fillna(0, inplace=True)



# BsmtQual, fill by "None"

train["BsmtQual"].fillna("None", inplace=True)

test["BsmtQual"].fillna("None", inplace=True)



# BsmtCond, fill by "None"

train["BsmtCond"].fillna("None", inplace=True)

test["BsmtCond"].fillna("None", inplace=True)



# BsmtExposure, fill by "None"

train["BsmtExposure"].fillna("None", inplace=True)

test["BsmtExposure"].fillna("None", inplace=True)



# BsmtFinType1, fill by "None"

train["BsmtFinType1"].fillna("None", inplace=True)

test["BsmtFinType1"].fillna("None", inplace=True)



# BsmtFinSF1, fill by 0

test["BsmtFinSF1"].fillna(0, inplace=True)



# BsmtFinType2, fill by "None"

train["BsmtFinType2"].fillna("None", inplace=True)

test["BsmtFinType2"].fillna("None", inplace=True)



# BsmtFinSF1, fill by 0

test["BsmtFinSF2"].fillna(0, inplace=True)



# BsmtUnfSF, fill by 0

test["BsmtUnfSF"].fillna(0, inplace=True)



# TotalBsmtSF

test["TotalBsmtSF"].fillna(0, inplace=True)



# Electrical, fill by 1st mode "SBrkr"

train["Electrical"].fillna("SBrkr", inplace=True)



# BsmtFullBath, fill by 0

test["BsmtFullBath"].fillna(0, inplace=True)



# BsmtHalfBath, fill by 0

test["BsmtHalfBath"].fillna(0, inplace=True)



# KitchenQual, fill by 1st mode "TA"

test["KitchenQual"].fillna("TA", inplace=True)



# Functional, fill by 1st mode "Typ"

test["Functional"].fillna("Typ", inplace=True)



# FireplaceQu, fill by "None"

train["FireplaceQu"].fillna("None", inplace=True)

test["FireplaceQu"].fillna("None", inplace=True)



# GarageType, fill by "None"

train["GarageType"].fillna("None", inplace=True)

test["GarageType"].fillna("None", inplace=True)



# GarageYrBlt, fill by 0

train["GarageYrBlt"].fillna(0, inplace=True)

test["GarageYrBlt"].fillna(0, inplace=True)



# GarageFinish, fill by "None"

train["GarageFinish"].fillna("None", inplace=True)

test["GarageFinish"].fillna("None", inplace=True)



# GarageCars, fill by 0

test["GarageCars"].fillna(0, inplace=True)



# GarageArea, fill by 0

test["GarageArea"].fillna(0, inplace=True)



# GarageQual, fill by "None"

train["GarageQual"].fillna("None", inplace=True)

test["GarageQual"].fillna("None", inplace=True)



# GarageCond, fill by "None"

train["GarageCond"].fillna("None", inplace=True)

test["GarageCond"].fillna("None", inplace=True)



# PoolQC, drop columns

train.drop("PoolQC", axis=1, inplace=True)

test.drop("PoolQC", axis=1, inplace=True)



# Fence, drop columns

train.drop("Fence", axis=1, inplace=True)

test.drop("Fence", axis=1, inplace=True)



# MiscFeature, drop columns

train.drop("MiscFeature", axis=1, inplace=True)

test.drop("MiscFeature", axis=1, inplace=True)



# SaleType, fill by 1st mode

test["SaleType"].fillna("WD", inplace=True)
# If use only like decision tree algorism, Cahge to numerical balues with OridinalEncoder



# pick up categorical columns

#col_df = pd.DataFrame({"col":train.dtypes.index, "dtype":train.dtypes.values})

#obj_col = col_df[col_df["dtype"]=="object"]["col"].values



#for i in obj_col:

#    ce_oe = ce.OrdinalEncoder(cols=i, handle_unknown="impute")

#    train = ce_oe.fit_transform(train)

#    test = ce_oe.transform(test)

#    train[i] = train[i].astype("int16")
# Delete variables with very few variations

train.drop("Condition2", axis=1, inplace=True)

test.drop("Condition2", axis=1, inplace=True)



train.drop("RoofMatl", axis=1, inplace=True)

test.drop("RoofMatl", axis=1, inplace=True)



train.drop("LowQualFinSF", axis=1, inplace=True)

test.drop("LowQualFinSF", axis=1, inplace=True)



# Year-old relationship

# With renovation ⇒ Make a flag

# Change the year to the difference from the year of sale, leave the year of renovation and erase the original year of construction

def remode_flg(x):

    if x["YearBuilt"]==x["YearRemodAdd"]:

        res = 1

    else:

        res = 0

    return res



train["Remod_flg"] = train.apply(remode_flg, axis=1)

test["Remod_flg"] = test.apply(remode_flg, axis=1)



train["YearRemodAdd"] = train["YrSold"] - train["YearRemodAdd"] # House

test["YearRemodAdd"] = test["YrSold"] - test["YearRemodAdd"]



train["GarageYrBlt"] = train["YrSold"] - train["GarageYrBlt"] # Garage

test["GarageYrBlt"] = test["YrSold"] - test["GarageYrBlt"]



train.drop(["YearBuilt", "YrSold"], axis=1, inplace=True)

test.drop(["YearBuilt", "YrSold"], axis=1, inplace=True)



# Total Basement are is duplicated information with BsmtFinSF1 and BsmtFinSF2 and BsmtUnfSF

train.drop("TotalBsmtSF", axis=1, inplace=True)

test.drop("TotalBsmtSF", axis=1, inplace=True)



# Full Bathrooms, only to total counts

train["FullBathrooms"] = train["BsmtFullBath"] + train["FullBath"]

train.drop(["BsmtFullBath", "FullBath"], axis=1, inplace=True)



test["FullBathrooms"] = test["BsmtFullBath"] + test["FullBath"]

test.drop(["BsmtFullBath", "FullBath"], axis=1, inplace=True)



# Half Bathrooms, only to total counts

train["HalfBathrooms"] = train["BsmtHalfBath"] + train["HalfBath"]

train.drop(["BsmtHalfBath", "HalfBath"], axis=1, inplace=True)



test["HalfBathrooms"] = test["BsmtHalfBath"] + test["HalfBath"]

test.drop(["BsmtHalfBath", "HalfBath"], axis=1, inplace=True)



# Rooms, total rooms minus bedrooms

train["Rooms"] = train["TotRmsAbvGrd"] - train["BedroomAbvGr"]

train.drop("TotRmsAbvGrd", axis=1, inplace=True)



test["Rooms"] = test["TotRmsAbvGrd"] - test["BedroomAbvGr"]

test.drop("TotRmsAbvGrd", axis=1, inplace=True)



# "EnclosedPorch", "3SsnPorch", "ScreenPorch", One indoor pouch in total

train["Porch"] = train['WoodDeckSF'] + train['OpenPorchSF'] + train["EnclosedPorch"] + train["3SsnPorch"] + train["ScreenPorch"]

test["Porch"] = test['WoodDeckSF'] + test['OpenPorchSF'] + test["EnclosedPorch"] + test["3SsnPorch"] + test["ScreenPorch"]



train.drop(["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"], axis=1, inplace=True)

test.drop(["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"], axis=1, inplace=True)



# PoolArea, change to binary signal

train["PoolArea"] = train["PoolArea"].apply(lambda x:1 if x>0 else 0)

test["PoolArea"] = test["PoolArea"].apply(lambda x:1 if x>0 else 0)



# FirePlace, change to binary signal

train['Fireplaces'] = train['Fireplaces'].apply(lambda x:1 if x>0 else 0)

test['Fireplaces'] = test['Fireplaces'].apply(lambda x:1 if x>0 else 0)



# Logarithm for size or distance

log_col = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", 

           "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea", "Porch"]

for c in log_col:

    train[c] = np.log(train[c]+1)

    test[c] = np.log(test[c]+1)
# Cahge to numerical balues with OnehotEncoder



# pick up categorical columns

col_df = pd.DataFrame({"col":train.dtypes.index, "dtype":train.dtypes.values})

obj_col = col_df[col_df["dtype"]=="object"]["col"].values





for i in obj_col:

    ce_oe = ce.OneHotEncoder(cols=i, handle_unknown="impute")

    train = ce_oe.fit_transform(train)

    test = ce_oe.transform(test)
train.head()
test.head()
# select features

X_train  = train

X_test = test



y_train = target
# y_train is changed to log

y_train = np.log10(y_train)
%%time

# Instance

tree = DecisionTreeRegressor(random_state=10)



# Bayes search

opt_tree = BayesSearchCV(

            tree,

            {"max_depth":(5,20),

             "max_leaf_nodes":(30,100)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_tree = opt_tree.fit(X_train, y_train)



print("Best score:{}".format(bs_tree.best_score_))

print("Best params:{}".format(bs_tree.best_params_))
%%time

# Cross validation

tree = DecisionTreeRegressor(max_depth=9, max_leaf_nodes=67, random_state=10)



print("Cross validation score")

cr_score = cross_val_score(tree, X_train, y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
%%time

# Instance

forest = RandomForestRegressor(random_state=10)



# Bayes search

opt_forest = BayesSearchCV(

            forest,

            {"max_depth":(5,40),

             "max_leaf_nodes":(30,100),

             "n_estimators":(50,100),

             "min_samples_split":(0.00001,1.0),

             "min_samples_leaf":(1,10)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_forest = opt_forest.fit(X_train, y_train)



print("Best score:{}".format(bs_forest.best_score_))

print("Best params:{}".format(bs_forest.best_params_))
%%time

# Cross validation

forest = RandomForestRegressor(n_estimators=100, max_depth=40, max_leaf_nodes=100, min_samples_leaf=1, min_samples_split=0.00001, random_state=10)



print("Cross validation score")

cr_score = cross_val_score(forest, X_train, y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
forest.fit(X_train, y_train)



importance = forest.feature_importances_



indices = np.argsort(importance)[::-1]



for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" %(f+1, 30, X_train.columns[indices[f]], importance[indices[f]]))
%%time



scores = []



for i in range(50,len(indices),5):

    forest = RandomForestRegressor()

    # select features

    sel_features = X_train.columns[indices[:i]]

    

    # train & val data split

    X_tra, X_val, y_tra, y_val = train_test_split(X_train[sel_features], y_train, test_size=0.2, random_state=0)

    

    # Fitting and prediction

    forest.fit(X_tra, y_tra)

    y_pred = forest.predict(X_val)

    

    # accuracy socre

    score = r2_score(y_true=y_val, y_pred=y_pred)

    scores.append(score)

    

# visualization

plt.figure(figsize=(20,6))



plt.plot(range(50,len(indices),5), scores)

plt.xlabel("Number of feature")

plt.ylabel("Accuracy score")
# Feature select from importance

sel_features_forest = X_train.columns[indices[:130]]
%%time

# Cross validation

forest = RandomForestRegressor(n_estimators=100, max_depth=40, max_leaf_nodes=100, min_samples_leaf=1, min_samples_split=0.00001, random_state=10)



print("Cross validation score")

cr_score = cross_val_score(forest, X_train[sel_features_forest], y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
%%time

# Instance

xgbr = xgb.XGBRegressor(random_state=10)



# Bayes search

opt_xgb = BayesSearchCV(

            xgbr,

            {"learning_rate":(0.001,0.99),

             "max_depth":(1,30),

             "subsample":(0.1,0.99),

             "colsample_bytree":(0.01,0.99)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_xgb = opt_xgb.fit(X_train, y_train)



print("Best score:{}".format(bs_xgb.best_score_))

print("Best params:{}".format(bs_xgb.best_params_))
%%time

# Cross validation

xgbr = xgb.XGBRegressor(learning_rate=0.1350, max_depth=3, subsample=0.7356, colsample_bytree=0.7682, random_state=10)



print("Cross validation score")

cr_score = cross_val_score(xgbr, X_train, y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
%%time

# Instance

lgbm = lgb.LGBMRegressor()



# Bayes search

opt_lgbm = BayesSearchCV(

            lgbm,

            {"num_leaves":(20,100),

             "n_estimators":(10,100),

             "learning_rate":(0.001,1),

             "max_depth":(1,30),

             "min_split_gain":(0,0.5),

             "min_child_weight":(0.001,0.1),

             "min_child_samples":(5,50),

             "subsample":(0.1,0.99),

             "colsample_bytree":(0.01,0.99)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_lgbm = opt_lgbm.fit(X_train, y_train)



print("Best score:{}".format(bs_lgbm.best_score_))

print("Best params:{}".format(bs_lgbm.best_params_))
%%time

# Cross validation

lgbm = lgb.LGBMRegressor(learning_rate=0.2660, max_depth=26, colsample_bytree=0.6411, min_child_samples=36, min_child_weight=0.09835, min_split_gain=0, n_estimators=100, num_leaves=25, subsample=0.1146)



print("Cross validation score")

cr_score = cross_val_score(lgbm, X_train, y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
lgbm.fit(X_train, y_train)



importance = lgbm.feature_importances_



indices = np.argsort(importance)[::-1]



for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" %(f+1, 30, X_train.columns[indices[f]], importance[indices[f]]))
%%time



scores = []



for i in range(50,len(indices),5):

    lgbm = lgb.LGBMRegressor()

    # select features

    sel_features = X_train.columns[indices[:i]]

    

    # train & val data split

    X_tra, X_val, y_tra, y_val = train_test_split(X_train[sel_features], y_train, test_size=0.2, random_state=0)

    

    # Fitting and prediction

    lgbm.fit(X_tra, y_tra)

    y_pred = lgbm.predict(X_val)

    

    # accuracy socre

    score = r2_score(y_true=y_val, y_pred=y_pred)

    scores.append(score)

    

# visualization

plt.figure(figsize=(20,6))



plt.plot(range(50,len(indices),5), scores)

plt.xlabel("Number of feature")

plt.ylabel("Accuracy score")
# Feature select from importance

sel_features_lgbm = X_train.columns[indices[:100]]
%%time

# Cross validation

lgbm = lgb.LGBMRegressor(learning_rate=0.2660, max_depth=26, colsample_bytree=0.6411, min_child_samples=36, min_child_weight=0.09835, min_split_gain=0, n_estimators=100, num_leaves=25, subsample=0.1146)



print("Cross validation score")

cr_score = cross_val_score(lgbm, X_train[sel_features_lgbm], y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
# Scaling features

sc = StandardScaler()

sc.fit(X_train)



X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
# Training and score

ridge = Ridge()



# Bayes search

opt_r = BayesSearchCV(

            ridge,

            {"alpha":(0.0001,1000)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_r = opt_r.fit(X_train_std, y_train)



print("Best score:{}".format(bs_r.best_score_))

print("Best params:{}".format(bs_r.best_params_))
# Cross validation

ridge = Ridge(alpha=270.2)



print("Cross validation score")

cr_score = cross_val_score(ridge, X_train_std, y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
# Training and score

lasso = Lasso()



# Bayes search

opt_l = BayesSearchCV(

            lasso,

            {"alpha":(0.00001,1)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_l = opt_l.fit(X_train_std, y_train)



print("Best score:{}".format(bs_l.best_score_))

print("Best params:{}".format(bs_l.best_params_))
# Cross validation

lasso = Lasso(alpha=0.00001)



print("Cross validation score")

cr_score = cross_val_score(lasso, X_train_std, y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
# Training and score

elas = ElasticNet()



# Bayes search

opt_e = BayesSearchCV(

            elas,

            {"alpha":(0.00001,1)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_e = opt_e.fit(X_train_std, y_train)



print("Best score:{}".format(bs_e.best_score_))

print("Best params:{}".format(bs_e.best_params_))
# Cross validation

elas = ElasticNet(alpha=0.00365)



print("Cross validation score")

cr_score = cross_val_score(elas, X_train_std, y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
%%time

estimators = [("tr", DecisionTreeRegressor(max_depth=9, max_leaf_nodes=67, random_state=10)),

              ("rf", RandomForestRegressor(n_estimators=100, max_depth=40, max_leaf_nodes=100, min_samples_leaf=1, min_samples_split=0.00001, random_state=10)),

              ("xg", xgb.XGBRegressor(learning_rate=0.1350, max_depth=3, subsample=0.7356, colsample_bytree=0.7682, random_state=10)),

              ("lg", lgb.LGBMRegressor(learning_rate=0.2660, max_depth=26, colsample_bytree=0.6411, min_child_samples=36, min_child_weight=0.09835, min_split_gain=0, n_estimators=100, num_leaves=25, subsample=0.1146)),

              ("ri", Ridge(alpha=5.91)), # best params with no scaling

              ("la", Lasso(alpha=0.00001)), # best params with no scaling

              ("el", ElasticNet(alpha=0.00069))] # best params with no scaling



sreg = StackingRegressor(estimators=estimators)



sreg.fit(X_train, y_train)



print("Score:{}".format(sreg.score(X_train, y_train)))

print("Params:{}".format(sreg.get_params))
%%time

print("Cross validation score")

cr_score = cross_val_score(sreg, X_train, y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
# prediction

ps_test = sreg.predict(X_test)



X_test["y:pseudo"] = ps_test
Ps_X_train = X_test.sample(int(len(X_train)/2))

Ps_X_train.reset_index(drop=True, inplace=True)



Ps_y_train = Ps_X_train["y:pseudo"]

Ps_X_train = Ps_X_train.drop("y:pseudo", axis=1)



n_X_train = pd.concat([X_train, Ps_X_train]).reset_index(drop=True)

n_y_train = pd.concat([y_train, Ps_y_train]).reset_index(drop=True)
%%time

# Instance

tree = DecisionTreeRegressor(random_state=10)



# Bayes search

opt_tree = BayesSearchCV(

            tree,

            {"max_depth":(5,20),

             "max_leaf_nodes":(30,100)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_tree = opt_tree.fit(n_X_train, n_y_train)



print("Best score:{}".format(bs_tree.best_score_))

print("Best params:{}".format(bs_tree.best_params_))
%%time

# Cross validation

tree = DecisionTreeRegressor(max_depth=10, max_leaf_nodes=72, random_state=10)



print("Cross validation score")

cr_score = cross_val_score(tree, n_X_train, n_y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
%%time

# Instance

forest = RandomForestRegressor(random_state=10)



# Bayes search

opt_forest = BayesSearchCV(

            forest,

            {"max_depth":(5,40),

             "max_leaf_nodes":(30,100),

             "n_estimators":(50,100),

             "min_samples_split":(0.00001,1.0),

             "min_samples_leaf":(1,10)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_forest = opt_forest.fit(n_X_train[sel_features_forest], n_y_train)



print("Best score:{}".format(bs_forest.best_score_))

print("Best params:{}".format(bs_forest.best_params_))
%%time

# Cross validation

forest = RandomForestRegressor(n_estimators=100, max_depth=40, max_leaf_nodes=100, min_samples_leaf=1, min_samples_split=0.00001, random_state=10)



print("Cross validation score")

cr_score = cross_val_score(forest, n_X_train[sel_features_forest], n_y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
%%time

# Instance

xgbr = xgb.XGBRegressor(random_state=10)



# Bayes search

opt_xgb = BayesSearchCV(

            xgbr,

            {"learning_rate":(0.001,0.99),

             "max_depth":(1,30),

             "subsample":(0.1,0.99),

             "colsample_bytree":(0.01,0.99)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_xgb = opt_xgb.fit(n_X_train, n_y_train)



print("Best score:{}".format(bs_xgb.best_score_))

print("Best params:{}".format(bs_xgb.best_params_))
%%time

# Cross validation

xgbr = xgb.XGBRegressor(learning_rate=0.07403, max_depth=211, subsample=0.7613, colsample_bytree=0.7985, random_state=10)



print("Cross validation score")

cr_score = cross_val_score(xgbr, n_X_train, n_y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
%%time

# Instance

lgbm = lgb.LGBMRegressor()



# Bayes search

opt_lgbm = BayesSearchCV(

            lgbm,

            {"num_leaves":(20,100),

             "n_estimators":(10,100),

             "learning_rate":(0.001,1),

             "max_depth":(1,30),

             "min_split_gain":(0,0.5),

             "min_child_weight":(0.001,0.1),

             "min_child_samples":(5,50),

             "subsample":(0.1,0.99),

             "colsample_bytree":(0.01,0.99)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_lgbm = opt_lgbm.fit(n_X_train[sel_features_lgbm], n_y_train)



print("Best score:{}".format(bs_lgbm.best_score_))

print("Best params:{}".format(bs_lgbm.best_params_))
%%time

# Cross validation

lgbm = lgb.LGBMRegressor(learning_rate=0.1297, max_depth=20, colsample_bytree=0.9295, min_child_samples=43, min_child_weight=0.0484, min_split_gain=0, n_estimators=77, num_leaves=22, subsample=0.1)



print("Cross validation score")

cr_score = cross_val_score(lgbm, n_X_train[sel_features_lgbm], n_y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
n_X_train_std = sc.transform(n_X_train)
%%time

# Training and score

ridge = Ridge()



# Bayes search

opt_r = BayesSearchCV(

            ridge,

            {"alpha":(0.0001,1000)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_r = opt_r.fit(n_X_train_std, n_y_train)



print("Best score:{}".format(bs_r.best_score_))

print("Best params:{}".format(bs_r.best_params_))
%%time

# Cross validation

ridge = Ridge(alpha=70.98)



print("Cross validation score")

cr_score = cross_val_score(ridge, n_X_train_std, n_y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
%%time

# Training and score

lasso = Lasso()



# Bayes search

opt_l = BayesSearchCV(

            lasso,

            {"alpha":(0.00001,1)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_l = opt_l.fit(n_X_train_std, n_y_train)



print("Best score:{}".format(bs_l.best_score_))

print("Best params:{}".format(bs_l.best_params_))
%%time

# Cross validation

lasso = Lasso(alpha=0.00012)



print("Cross validation score")

cr_score = cross_val_score(lasso, n_X_train_std, n_y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
%%time

# Training and score

elas = ElasticNet()



# Bayes search

opt_e = BayesSearchCV(

            elas,

            {"alpha":(0.00001,1)},

            n_iter = 50,

            cv = 5,

            n_jobs = -1)



# Fitting

bs_e = opt_e.fit(n_X_train_std, n_y_train)



print("Best score:{}".format(bs_e.best_score_))

print("Best params:{}".format(bs_e.best_params_))
%%time

# Cross validation

elas = ElasticNet(alpha=0.00011)



print("Cross validation score")

cr_score = cross_val_score(elas, n_X_train_std, n_y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
%%time

estimators = [("tr", DecisionTreeRegressor(max_depth=10, max_leaf_nodes=72, random_state=10)),

              ("rf", RandomForestRegressor(n_estimators=100, max_depth=40, max_leaf_nodes=100, min_samples_leaf=1, min_samples_split=0.00001, random_state=10)),

              ("xg", xgb.XGBRegressor(learning_rate=0.07403, max_depth=211, subsample=0.7613, colsample_bytree=0.7985, random_state=10)),

              ("lg", lgb.LGBMRegressor(learning_rate=0.1297, max_depth=20, colsample_bytree=0.9295, min_child_samples=43, min_child_weight=0.0484, min_split_gain=0, n_estimators=77, num_leaves=22, subsample=0.1)),

              ("ri", Ridge(alpha=0.0001)), # best params with no scaling

              ("la", Lasso(alpha=0.00001)), # best params with no scaling

              ("el", ElasticNet(alpha=0.00001))] # best params with no scaling



sreg = StackingRegressor(estimators=estimators)



sreg.fit(n_X_train, n_y_train)



print("Score:{}".format(sreg.score(n_X_train, n_y_train)))

print("Params:{}".format(sreg.get_params))
print("Cross validation score")

cr_score = cross_val_score(sreg, n_X_train, n_y_train, cv=10)

print(cr_score)

print("Ave_score:{} ± {}".format(cr_score.mean(), cr_score.std()))
X_test.drop("y:pseudo", axis=1, inplace=True)

X_test_std = sc.transform(X_test)
# prediction

y_pred_tree = 10**(bs_tree.predict(X_test))

y_pred_forest = 10**(bs_forest.predict(X_test[sel_features_forest]))

y_pred_xgb = 10**(bs_xgb.predict(X_test))

y_pred_lgbm = 10**(bs_lgbm.predict(X_test[sel_features_lgbm]))

y_pred_r = 10**(bs_r.predict(X_test_std))

y_pred_l = 10**(bs_l.predict(X_test_std))

y_pred_sreg = 10**(sreg.predict(X_test))
# XGBoost submission

submit_xgb = sample.copy()

submit_xgb["SalePrice"] = y_pred_xgb
# LightGBM submission

submit_lgbm = sample.copy()

submit_lgbm["SalePrice"] = y_pred_lgbm
# stackmodel submission

submit_sreg = sample.copy()

submit_sreg["SalePrice"] = y_pred_sreg
# ems submission

submit_ems = sample.copy()

submit_ems["SalePrice"] = y_pred_tree*0.05 + y_pred_forest*0.1 + y_pred_xgb*0.3 + y_pred_lgbm*0.2 + y_pred_r*0.05 + y_pred_l*0.05 + y_pred_sreg*0.25
# Submission

submit_xgb.to_csv('xgb_submission.csv', index=False)

submit_lgbm.to_csv('lgbm_submission.csv', index=False)

submit_sreg.to_csv('stack_submission.csv', index=False)

submit_ems.to_csv('ems_submission.csv', index=False)

print("Your submission was successfully saved!")
submit_ems
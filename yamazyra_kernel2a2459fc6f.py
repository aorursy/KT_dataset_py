# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from datetime import datetime

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p,inv_boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# ファイルのインポート

df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df_test  = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

df_sample  = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

df_train.head()
# データの統合

train_features = df_train.copy()

test_features = df_test.copy()



train_features.drop(['Id'], axis=1, inplace=True)

test_features.drop(['Id'], axis=1, inplace=True)



train_features = train_features.drop(['SalePrice'], axis=1)

features = pd.concat([train_features, test_features]).reset_index(drop=True)

print("-1-------------")

print("元データの形式:(合体後、訓練、テスト)")

print(features.shape, train_features.shape, test_features.shape)



# 目的変数 

# SalePriceの状況を見る

print("-2-------------")

print("SalePriceの確認")

print(df_train["SalePrice"].describe())

y = df_train['SalePrice']

plt.figure(1); plt.title('Normal')

sns.distplot(y, kde=False, fit=stats.norm)

plt.figure(2); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=stats.lognorm)



# いったん、logスケールに変換することにする

y = np.log1p(df_train["SalePrice"])

#y_max = boxcox_normmax(y + 1)

#y = boxcox1p(y, y_max)



print("-3-------------")

# 欠損値の補正（特別なやつ）

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

print("-5-------------")

# 欠損値の補正（数値）

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

        features[i] = features[i].fillna(0)

print("-6-------------")

# 偏りのある数値データの補正

skew_features = features[numerics].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))

    

print("-7-------------")

# 新規データの追加(数値系)

features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])

print("-7.5-------------")

# 非数値の整形

features['MSSubClass'] = features['MSSubClass'].apply(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)



features['Functional'] = features['Functional'].fillna('Typ')

features['Electrical'] = features['Electrical'].fillna("SBrkr")

features['KitchenQual'] = features['KitchenQual'].fillna("TA")

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])



features["PoolQC"] = features["PoolQC"].fillna("None")



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')

print("-4-------------")

# 欠損値の補正（非数値）

for i in features.columns:

    if features[i].dtype == object:

        features[i] = features[i].fillna('None')







print("-8-------------")

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

print("-9-------------")

# 不要なデータの削除

features = features.drop(['Utilities', 'Street', 'PoolQC'], axis=1)





features = pd.get_dummies(features).reset_index(drop=True)

print("-10-------------")

# 分類用データの作成

X_train = features.iloc[:len(y), :]

X_test = features.iloc[len(y):, :]

print("-11-------------")

# 外れ値をオミット

outliers = [30, 88, 462, 631, 1322]

X_train = X_train.drop(X_train.index[outliers])

y = y.drop(y.index[outliers])

print("-12-------------")

# 過学習なパラメータをオミット

overfit = []

for i in X_train.columns:

    counts = X_train[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X_train) * 100 > 99.94:

        overfit.append(i)



overfit = list(overfit)



X_train = X_train.drop(overfit, axis=1)

X_test = X_test.drop(overfit, axis=1)

print("-13-------------")

print("特徴量操作後のデータの形式:(訓練、テスト,元の訓練、元のテスト)")

print(X_train.shape, X_test.shape, train_features.shape, test_features.shape)
# 分類の精度用関数設定

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X_train):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
# 分類器の作成

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)   

lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )

xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective="reg:squarederror", nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
# 交差検証

print("START CV")

score = cv_rmse(ridge)

print("\nRIDGE: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(lasso)

print("\nLASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(elasticnet)

print("\nelastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(svr)

print("\nSVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(lightgbm)

print("\nlightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(gbr)

print("\ngbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(xgboost)

print("\nxgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



# 学習

print('START Fit')



print('----stack_gen')

stack_gen_model = stack_gen.fit(np.array(X_train), np.array(y))



print('----elasticnet')

elastic_model_full_data = elasticnet.fit(X_train, y)



print('----Lasso')

lasso_model_full_data = lasso.fit(X_train, y)



print('----Ridge')

ridge_model_full_data = ridge.fit(X_train, y)



print('----Svr')

svr_model_full_data = svr.fit(X_train, y)



print('----GradientBoosting')

gbr_model_full_data = gbr.fit(X_train, y)



print('----xgboost')

xgb_model_full_data = xgboost.fit(X_train, y)



print('----lightgbm')

lgb_model_full_data = lightgbm.fit(X_train, y)
def blend_models_predict(X):

    return ((0.1 * elastic_model_full_data.predict(X)) + \

            (0.05 * lasso_model_full_data.predict(X)) + \

            (0.1 * ridge_model_full_data.predict(X)) + \

            (0.1 * svr_model_full_data.predict(X)) + \

            (0.1 * gbr_model_full_data.predict(X)) + \

            (0.15 * xgb_model_full_data.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.3 * stack_gen_model.predict(np.array(X))))
def inv_sales_boxcox(Y,Ymax):

    return np.round(inv_boxcox1p(Y,Ymax))



def inv_sales_log(Y):

    return np.round(np.expm1(Y))
print('RMSLE score on train data:')

print(rmsle(y, blend_models_predict(X_train)))

df_sub = X_train.copy()

#df_sub["predict"] = inv_sales_boxcox(blend_models_predict(X_train),y_max)

df_sub["predict"] = inv_sales_log(blend_models_predict(X_train))

df_sub["SalePrice"] = df_train["SalePrice"] 



df_sub[["SalePrice","predict"]]
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission["SalePrice"]=inv_sales_log(blend_models_predict(X_test))

#submission["SalePrice"]=inv_sales_boxcox(blend_models_predict(X_test),y_max)

submission
print('Blend with Top Kernals submissions')

sub_1 = pd.read_csv('../input/top-10-0-10943-stacking-mice-and-brutal-force/House_Prices_submit.csv')

sub_2 = pd.read_csv('../input/hybrid-svm-benchmark-approach-0-11180-lb-top-2/hybrid_solution.csv')

sub_3 = pd.read_csv('../input/lasso-model-for-regression-problem/lasso_sol.csv')

#sub_4 = pd.read_csv('../input/all-you-need-is-pca-lb-0-11421-top-4/submission.csv')

# sub_5 = pd.read_csv('../input/house-prices-solution-0-107-lb/submission.csv') # fork my kernel again)



submission.iloc[:,1] = np.floor((0.15 * submission["SalePrice"]) + 

                                (0.20 * sub_1.iloc[:,1]) + 

                                (0.30 * sub_2.iloc[:,1]) + 

                                (0.25 * sub_3.iloc[:,1]) 

                                #(0.15 * sub_4.iloc[:,1])  

                                #(0.1 * sub_5.iloc[:,1])

                                )





# From https://www.kaggle.com/agehsbarg/top-10-0-10943-stacking-mice-and-brutal-force

# Brutal approach to deal with predictions close to outer range 

q1 = submission['SalePrice'].quantile(0.0042)

q2 = submission['SalePrice'].quantile(0.99)



submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

# 



submission.to_csv("submission_blending1.csv", index=False)

print('Save submission', datetime.now(),)
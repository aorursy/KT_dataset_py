from copy import deepcopy

import random
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso, ElasticNet

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from xgboost import XGBRegressor



import pandas as pd

import numpy as np

from scipy.stats import skew
%matplotlib inline
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
df_train_raw = pd.read_csv('../input/train.csv')

df_train_raw.head(3)
df_train_raw.shape
df_test_raw = pd.read_csv('../input/test.csv')

df_test_raw.head()

ids_test = df_test_raw["Id"]
y_name = 'SalePrice'
df_train_raw[y_name] = np.log(df_train_raw[y_name])
cont_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 

             'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 

             'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 

            '3SsnPorch', 'ScreenPorch', 'PoolArea']
scaler = StandardScaler()

df_num = df_train_raw[cont_cols].copy()

df_num[cont_cols] = scaler.fit_transform(df_num[cont_cols])
df_num[cont_cols[:10]].boxplot(figsize=(14, 10))
df_num[cont_cols[10:]].boxplot(figsize=(14, 10))
cut = {'LotFrontage': 5, 'LotArea': 5.5, 'BsmtFinSF1': 10, 'BsmtFinSF2': 5.5, 

       'TotalBsmtSF': 10, '1stFlrSF': 5.5, 'GrLivArea': 5, 'WoodDeckSF': 5.1, 

       'OpenPorchSF': 5, 'EnclosedPorch': 5.1}



indexes = list()

for col in cut:

    indexes += list(df_num[df_num[col]>cut[col]].index)

indexes = set(indexes)

print(df_train_raw.shape)

df_train_raw = df_train_raw.drop(indexes)

print(df_train_raw.shape)
len_train = df_train_raw.shape[0]

df_raw = pd.concat([df_train_raw, df_test_raw], sort=False)

df_raw = df_raw.reset_index(drop=True)

df_raw = df_raw.drop('Id', 1)
(df_raw.isnull().sum() / df_raw.shape[0]).sort_values(ascending=False).head(10)
df = df_raw.copy()
df['MSSubClass'] = df['MSSubClass'].astype(str)

df['MoSold'] = df['MoSold'].astype(str)
nanvars = {'Alley': 'None', 'BsmtQual': 'None', 'BsmtCond': 'None', 'BsmtExposure': 'None', 

           'BsmtFinType1': 'None', 'BsmtFinType2': 'None', 'FireplaceQu': 'None', 

           'GarageType': 'None', 'GarageFinish': 'None', 'GarageQual': 'None', 

           'GarageCond': 'None', 'PoolQC': 'None', 'Fence': 'None', 'MiscFeature': 'None',

           'Utilities': 'ELO', 'MasVnrType': 'None', 'MasVnrArea': 0, 'Functional': 'Typ',

           'BsmtFinSF1': 0, 'BsmtFinSF2': 0, 'BsmtUnfSF': 0,'TotalBsmtSF': 0, 'BsmtFullBath': 0, 

           'BsmtHalfBath': 0, 'MSSubClass': 'None', 'GarageCars': 0, 'GarageArea': 0

}



nanmode = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 

           'GarageYrBlt']

for c in df.columns:

    if df[c].isnull().sum() == 0 or c == y_name: 

        continue

    df[c+'_na'] = df[c].isnull()

    if c in nanvars:

        df[c] = df[c].fillna(nanvars[c])

    elif c in nanmode:

        df[c] = df[c].fillna(df[c].mode()[0])



for c in ['LotFrontage']:

    df[c+'_na'] = df[c].isnull()

df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

df['TotalSF2'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + 

                  df['1stFlrSF'] + df['2ndFlrSF'])

df['TotalBathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + 

                        df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + 

                      df['EnclosedPorch'] + df['ScreenPorch'] + 

                      df['WoodDeckSF'])



df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df['HasBsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
def get_high_corr(df, threshold):

    corr = df.corr()

    showcols = list()

    for y_col, row in corr.iterrows():

        for x_col, val in zip(row.index, row):

            if x_col != y_col and val >= threshold:

                showcols.append(x_col)

                showcols.append(y_col)

    return list(set(showcols))
showcols = get_high_corr(df, 0.99999)

sns.set()

plt.figure(figsize=(16, 12))

sns.heatmap(df[showcols].corr())
todel = ['Exterior1st_na', 'BsmtFinSF1_na', 'BsmtFinSF2_na', 'BsmtUnfSF_na', 

         'BsmtHalfBath_na', 'Exterior2nd_na', 'GarageYrBlt_na', 

         'GarageFinish_na', 'GarageCond_na', 'GarageCars_na']

df = df.drop(todel, 1)
plt.figure(figsize=(16, 10))

sns.heatmap(df[[c for c in showcols if c not in todel]].corr())
gradcats = {

    'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'], 

    'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'],

    'Utilities': ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'], 

    'LandSlope': ['Gtl', 'Mod', 'Sev'],

    'ExterQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'], 

    'ExterCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],

    'BsmtQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], 

    'BsmtCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'],

    'BsmtExposure': ['Gd', 'Av', 'Mn', 'No', 'None'], 

    'BsmtFinType1': ['GLQ', 'ALQ', 'BLQ','Rec','LwQ', 'Unf', 'None'],

    'BsmtFinType2': ['GLQ', 'ALQ', 'BLQ','Rec','LwQ', 'Unf', 'None'], 

    'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],

    'KitchenQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'], 

    'FireplaceQu': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'],

    'GarageFinish': ['Fin', 'RFn', 'Unf', 'None'], 

    'GarageQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'],

    'GarageCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], 

    'PoolQC': ['Ex', 'Gd', 'TA', 'Fa', 'None'],

    'Fence': ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'None'],

    'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']

}

for c in df.columns:

    if df[c].dtype == 'object':

        if c in gradcats:

            subs = list()

            for val in gradcats[c]:

                if val in df[c].values:

                    subs.append(val)

            df[c] = df[c].astype('category')

            df[c] = df[c].cat.reorder_categories(subs)

            df[c] = df[c].cat.codes

for c in df.columns:

    if c == y_name or df[c].dtype == 'object':

        continue

    df[c + '_sqrt'] = np.sqrt(df[c])

    df[c + '_times_2'] = df[c] ** 2
df = pd.get_dummies(df) 
numeric_feats = df.dtypes[df.dtypes != 'object'].index

numeric_feats = [c for c in numeric_feats if df[c].nunique() > 2]

skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)



skewness = pd.DataFrame({'Skew' :skewed_feats})

print(skewness.head())

print()

print(skewness.tail())
skew_val = 1.35

print(f'Number of columns to transform: {skewness[abs(skewness) > skew_val].notnull().sum()[0]}')
for c in skewness.index:

    if c != y_name and abs(skewness.loc[c][0]) > skew_val:

        df[c + '_log'] = np.log1p(df[c])

#         df[c] = np.log1p(df[c])
def print_cv_result(cv: list):

    print(f'CV mean: {round(np.mean(cv), 6)}, CV std: {round(np.std(cv), 6)}')



def print_feature_importance(columns, values, abs_threshold=0.0):

    fi = {name: val for name, val in zip(columns, values)}

    for k in sorted(fi, key=fi.get, reverse=True):

        if abs(fi[k]) >= abs_threshold:

            print(k.rjust(30), fi[k])
X_train = df[:len_train].drop(y_name, 1)

y_train = df[:len_train][y_name]
def scorer(estimator, X, y):

    return np.sqrt(mean_squared_error(y, estimator.predict(X)))
xgb = XGBRegressor(

    random_state=1, n_estimators=100, verbosity=1, objective='reg:squarederror', 

    **{'subsample': 0.5308, 'reg_lambda': 1.051, 'min_child_weight': 1, 'max_depth': 3, 

       'colsample_bytree': 0.4, 'eta': 0.3, 'gamma': 0})

cv = cross_val_score(xgb, X_train, y_train, cv=4, scoring=scorer)

print_cv_result(cv)
xgb.fit(X_train, y_train)

print_feature_importance(X_train.columns, xgb.feature_importances_, abs_threshold=0.01)
en = make_pipeline(RobustScaler(), ElasticNet(random_state=1, alpha=0.00065, max_iter=3000, l1_ratio=0.7))

cv = cross_val_score(en, X_train, y_train, cv=4, scoring=scorer)

print_cv_result(cv)
en.fit(X_train, y_train)

print_feature_importance(X_train.columns, en.steps[1][1].coef_, abs_threshold=0.05)
class BlendingModel():

    def __init__(self, weights: dict, models: dict):

        if list(weights.keys()) != list(models.keys()):

            raise ValueError('Weights and models must have the same keys')

        self.weights = deepcopy(weights)

        self.models = deepcopy(models)

        

    def fit(self, X, y, verbose=True):

        for m_name in self.models:

            if verbose:

                print(f'fitting model {m_name}')

            self.models[m_name].fit(X, y)

    

    def predict(self, X):

        return sum(self.weights[m_name] * self.models[m_name].predict(X) 

                   for m_name in self.models)

        
def find_weights_step(models: dict, df, y_name, step=0.05):

    if len(models.keys()) != 2:

        raise ValueError('find_weights_step() Only works if there are two models in dict')

    best_weights = float('inf')

    best_score = float('inf')

    

    df_tmp = df.sample(random_state=5, frac=1)

    df_train = df_tmp[:1100]

    df_val = df_tmp[1100:]

    X_train, y_train = df_train.drop(y_name, 1), df_train[y_name]

    X_val, y_val = df_val.drop(y_name, 1), df_val[y_name]

        

    

    weights = dict()

    models_keys = list(models.keys())

    weights[models_keys[0]] = 1.

    weights[models_keys[1]] = 0.

    

    models_ = deepcopy(models)

    for m in models_:

        models_[m].fit(X_train, y_train)

    

    while weights[models_keys[0]] > 0:

        pred_stack = sum(weights[m_name] * models_[m_name].predict(X_val) 

                         for m_name in models_)

        score = np.sqrt(mean_squared_error(y_val, pred_stack))

        

        print('score:', score, 'at', weights)

        if score < best_score:

            best_weights = deepcopy(weights)

            best_score = score

            

        weights[models_keys[0]] -= step

        weights[models_keys[1]] += step

    print('best validation score:', best_score,  'at', best_weights)

    return best_weights
models = {'xgb': xgb, 'en': en}
weights = find_weights_step(models, df[:len_train], y_name, step=0.1)
weights = {'xgb': 0.3, 'en': 0.7}
X = df[:len_train].drop(y_name, 1)

y = df[:len_train][y_name]

X_test = df[len_train:].drop('SalePrice', 1)
xgb_final = XGBRegressor(

    random_state=1, n_estimators=2100, verbosity=1, objective='reg:squarederror', nthread=-1,

    **{'subsample': 0.5308, 'reg_lambda': 1.051, 'min_child_weight': 1, 'max_depth': 3,

       'colsample_bytree': 0.5, 'eta': 0.4, 'gamma': 0})

xgb_final.fit(X, y)
en_final = make_pipeline(RobustScaler(), 

                         ElasticNet(random_state=1, alpha=0.00065, max_iter=3000))

en_final.fit(X, y)
models = {'xgb': xgb_final, 'en': en_final}

blending_final = BlendingModel(

    weights=weights,

    models=models

)
SalePrice = np.exp(blending_final.predict(X_test))



out_df = pd.DataFrame({'Id': ids_test, "SalePrice": SalePrice})

out_df.to_csv('submission.csv', index=False)
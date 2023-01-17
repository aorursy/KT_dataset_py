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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

print(f'SHAPE  rows, cols\ntrain: {train.shape}\ntest: {test.shape}')
def match_list(lista1, lista2):
    for i in lista1:
        if i not in lista2:
            print(f'{i}: not exists list 2')
    for z in lista2:
        if z not in lista1:
            print(f'{z}: not exists list 1')

match_list(train.columns, test.columns)
# Concatenate the train and test
X_full = pd.concat([train.drop("SalePrice", axis=1), test]) 
y_full = train[['SalePrice']]

print(f'SHAPE  rows, cols\nX_full: {X_full.shape}\ny_full: {y_full.shape}')
# Missing values
miss_values = X_full.isna().sum()
print(f'Total of missing values: {miss_values.sum()}')
cols_miss = miss_values[miss_values.values > 0].index
print(f'Total of Columns missing values: {len(cols_miss)}\nList columns missing values:\n{cols_miss}')

miss_values = X_full.isna().sum().sort_values(ascending = False).head(22).reset_index()
miss_values.rename(columns={'index': 'features', 0: 'miss_values'}, inplace=True)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

with plt.style.context('Solarize_Light2'):
    plt.figure(figsize=(14,5))
    color = sns.dark_palette("deeppink", reverse=True, n_colors=22)
    plt.bar(miss_values['features'], round(miss_values.miss_values*100 / len(X_full), 1), color=color, width=0.9)
    xs = list(range(0,120,20))
    ys = list(map(lambda x: str(x)+'%', xs))
    plt.yticks(xs, ys)
    plt.xticks(rotation=90)
    plt.title('Columns Missing Values TOP: 22', color='#073642', style='italic')
    plt.xlabel('Features')
    plt.ylabel('% Of Missing Values')
plt.show()

miss_drop_cols = miss_values.loc[0:5].features.values.tolist() # add list columns % top 6 missing values
print('Total of Features Full: {}'.format(len(X_full.columns)))
X_full.drop(columns=miss_drop_cols, inplace=True)
print('Total of Features Full drop missing: {}'.format(len(X_full.columns)))
# Categorical Features
cat_features = X_full.select_dtypes(include='object').copy()
cat_features['MSSubClass'] = X_full['MSSubClass'].apply(str)

# Numeric Features
numeric_features = X_full.select_dtypes(exclude='object').copy()
numeric_features.drop(columns=['MSSubClass'], inplace=True)
#drop(['MSSubClass'])

print('Total of features category: {}'.format(len(cat_features.columns)))
print('Total of features numeric: {}'.format(len(numeric_features.columns)))
#cat_features.nunique().sort_values(ascending = False).head(3)
with plt.style.context('Solarize_Light2'):
    fig = plt.figure(figsize=(18,24))
    for index, i in enumerate(numeric_features.columns[1:]):
        plt.subplot(9,4,index+1)
        sns.distplot(numeric_features[i].dropna(), color='#268bd2', kde=False, hist_kws={'alpha': .8}) 
    
    fig.suptitle('Histograms Features Numerical', y=1.02, fontsize=18, color='#002b36',  weight='bold')
    fig.tight_layout(pad=1.0)
numeric_overfit = []
for i in numeric_features.columns:
    z = numeric_features[i].value_counts(normalize=True)*100
    if z.iloc[0] >= 85:
        numeric_overfit.append(i)

print(f"Numerical Features with > 85% of the same value: {numeric_overfit}")
with plt.style.context('Solarize_Light2'):
    fig = plt.figure(figsize=(18,20))
    for index, i in enumerate(numeric_features.drop(columns=numeric_overfit).columns[1:]):
        plt.subplot(7,4,index+1)
        sns.regplot(x=i, y='SalePrice', data=train, color='#268bd2')
        p = plt.xticks()[0]
        var = p[-1]- p[-2]
        plt.xlim(xmax=train[i].max()+var/4)
    
    fig.suptitle('Regression Features Numerical', y=1.02, fontsize=18, color='#002b36',  weight='bold')
    fig.tight_layout(pad=1.0)
b = numeric_features.drop(columns=numeric_overfit)
b = b.drop(b.columns[[0]], axis=1)

df_corr = b
df_corr['SalePrice'] = train['SalePrice']
corr = df_corr.corr()

with plt.style.context('Solarize_Light2'):
    fig, (heat1, heat2) = plt.subplots(2, 1, figsize=(16, 20), sharex=True, gridspec_kw={'hspace': .08})
    sns.heatmap(round(corr,1), linewidth=0.5, annot=True, ax=heat1)
    sns.heatmap(round(corr,1), mask = corr <=0.8, linewidth=0.5, annot=True, ax=heat2)
    heat1.set_title('Heatmap 1: Correlated all', color='#073642', style='italic')
    heat2.set_title('Heatmap 2: Highly Correlated variables > 0.8', color='#073642', style='italic')
    fig.suptitle('Heatmaps Features Numerical', x=0.17, y=0.92, fontsize=18, color='#002b36',  weight='bold')
    plt.show()
# Missing values features numerical
from sklearn.impute import KNNImputer

print('Total of missing values features numerical: {}'.format(numeric_features.isna().sum().sum()))
print('Filling missing values using the k-Nearest Neighbors....')
cols_name1 = list(numeric_features.columns)
imputer = KNNImputer(n_neighbors=2, weights="distance")
imp_knn = imputer.fit_transform(numeric_features)
numeric_features = pd.DataFrame(imp_knn, columns=cols_name1, index=numeric_features.index)

print('Total of missing values features numerical transform k-Nearest Neighbors: {}'.format(numeric_features.isna().sum().sum()))
h = []
for c in corr.columns:
    high = corr[c].mask(corr[c] < 0.8).dropna().index
    if len(high) > 1:
        h.append(high[0]+' '+'and'+' '+high[1])

print('Highly Correlated variables:')
for o in set(h):
    print(o)
cat_features['YrSold'] = numeric_features['YrSold'].astype(str)
cat_features['MoSold'] = numeric_features['MoSold'].astype(str)
numeric_features.drop(columns=['YrSold', 'MoSold'], inplace=True)
numeric_features.drop(columns=numeric_overfit, inplace=True)
with plt.style.context('Solarize_Light2'):
    fig = plt.figure(figsize=(18,30))
    for index, i in enumerate(cat_features.columns):
        plt.subplot(11,4,index+1)
        sns.countplot(x=cat_features[i])
        plt.xticks(rotation=90)
    
    fig.suptitle('Counts Features Categorical', y=1.02, fontsize=18, color='#002b36',  weight='bold')
    fig.tight_layout(pad=1.0)
print('Total of missing values features categorical: {}'.format(cat_features.isna().sum().sum()))
# Features missing fill 'NA'
miss_na = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond',
           'BsmtFinType1','BsmtFinType2', 'BsmtExposure', 'GarageFinish']

for i in miss_na:
    cat_features[i] = cat_features[i].fillna('NA')

# Features missing fill 'mode'
for i in cat_features.columns:
    if i not in miss_na:
        cat_features[i] = cat_features[i].fillna(cat_features[i].mode()[0])

print('Total of missing values features categorical transform: {}'.format(cat_features.isna().sum().sum()))
ord_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond']
fin_col = ['BsmtFinType1','BsmtFinType2']

ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}
fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}
expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}

lotshape_map = {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0}
landslope_map = {'Gtl': 2, 'Mod': 1, 'Sev': 0}
garagefinish_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}
paved_map = {'Y': 2, 'P': 1, 'N': 0}


for col in ord_col:
    cat_features[col] = cat_features[col].map(ordinal_map)
    

for col in fin_col:
    cat_features[col] = cat_features[col].map(fintype_map)

cat_features['BsmtExposure'] = cat_features['BsmtExposure'].map(expose_map)
cat_features['LotShape'] = cat_features['LotShape'].map(lotshape_map)
cat_features['LandSlope'] = cat_features['LandSlope'].map(landslope_map)
cat_features['GarageFinish'] = cat_features['GarageFinish'].map(garagefinish_map)
cat_features['PavedDrive'] = cat_features['PavedDrive'].map(paved_map)

cat_overfit = []
for i in cat_features.columns:
    z = cat_features[i].value_counts(normalize=True)*100
    if z.iloc[0] > 96:
        cat_overfit.append(i)

print(f"Categorical Features with > 96% of the same value: {cat_overfit}")
cat_features.drop(columns=cat_overfit, inplace=True)

numeric_features['TotalSF'] = numeric_features['TotalBsmtSF'] + numeric_features['1stFlrSF'] + numeric_features['2ndFlrSF']
numeric_features['TotalBath'] = numeric_features['FullBath'] + numeric_features['HalfBath'] 
numeric_features['TotalPorch'] = numeric_features['OpenPorchSF'] + numeric_features['EnclosedPorch']
# Merge features numerical anda categorical
features = pd.concat([numeric_features, cat_features], axis=1)
features.drop(columns='Id', inplace=True)
features.shape, X_full.shape
y_target = np.log1p(y_full) # transform y log to target predict
with plt.style.context('Solarize_Light2'):
    fig, (hist, hist_log) = plt.subplots(1, 2, figsize=(16, 4))
    sns.distplot(y_full, color='#268bd2', ax=hist)
    sns.distplot(y_target, color='#268bd2', ax=hist_log)
    hist.set_title('SalePrice', color='#073642', style='italic')
    hist_log.set_title('Log(SalePrice)', color='#073642', style='italic')
    
    fig.suptitle('Histograms', y=1.02, fontsize=18, color='#002b36',  weight='bold')
    plt.show()
# OneHotEncoder
features = pd.get_dummies(features)

# Split train and test
X_train_new = features.iloc[:len(train), :]
X_test_new = features.iloc[len(X_train_new):, :]

print('SHAPE  rows, cols\ntrain_new: {}\ntest_new: {}'.format(X_train_new.shape, X_test_new.shape))
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

# ExtraTrees
extra_model = ExtraTreesRegressor(max_depth=200, n_estimators=570, random_state=1)

# GradientBoosting
grad_model = GradientBoostingRegressor(n_estimators=2900, learning_rate=0.0161, max_depth=3,
                                       max_features='sqrt', min_samples_leaf=17, loss='huber', random_state=1)
hist_model = HistGradientBoostingRegressor(min_samples_leaf=40, max_depth=3, max_iter=225, learning_rate=0.15,
                                           loss='least_absolute_deviation', random_state=1)
# xgboost
xgboost_model = XGBRegressor(learning_rate=0.0139, n_estimators=2000, max_depth=4, min_child_weight=0,
                             subsample=0.7968, colsample_bytree=0.4064, nthread=-1, scale_pos_weight=2,
                             seed=42, random_state=1)
# lightgbm
lgbm_model = LGBMRegressor(objective='regression', n_estimators=6500, num_leaves=10, learning_rate=0.005,
                           max_bin=163, bagging_fraction=0.85, n_jobs=-1, bagging_seed=42, 
                           feature_fraction_seed=42, bagging_freq=7, feature_fraction=0.1294, 
                           min_data_in_leaf=8, random_state=1)
# LassoCV
lasso_model = LassoCV(n_alphas=150, max_iter=1e4, random_state=1)

# ElasticNetCV
elasticnet_model = ElasticNetCV(n_alphas=150, max_iter=1e4, l1_ratio=1.15, random_state=1)

# Huber
huber_model = HuberRegressor(max_iter=2000)

# SVR
rbf_model = SVR(kernel='rbf', C=21, epsilon=0.0099, gamma=0.00017, tol=0.000121)
# Transformer
transformer = QuantileTransformer(output_distribution='normal')

# Models
extratree = make_pipeline(transformer, extra_model)
grad = make_pipeline(transformer, grad_model)
hist = make_pipeline(transformer, hist_model)
xgboost = make_pipeline(transformer, xgboost_model)
lgbm = make_pipeline(transformer, lgbm_model)
lasso = make_pipeline(transformer, lasso_model)
elasticnet = make_pipeline(transformer, elasticnet_model)
huber = make_pipeline(transformer, huber_model)
svr = make_pipeline(RobustScaler(), rbf_model)

models = [('ExtraTrees', extratree),
          ('GradientBoosting', grad),
          ('HistGradientBoosting', hist),
          ('XGBoost', xgboost), 
          ('LightGBM', lgbm),
          ('LassoCV', lasso),
          ('ElasticNetCV', elasticnet),
          ('Huber', huber),
          ('SVR', svr)]
def storm_model(x, y, models, cv, scoring):
    df_evaluation = pd.DataFrame()
    df_predict = pd.DataFrame()
    row_index = 0
    for name, model in models:
        # score
        scores = cross_validate(model, np.array(x), np.array(y).ravel(), cv=cv, scoring=scoring, n_jobs=-1, verbose=0)
        # predict
        y_pred = cross_val_predict(model, np.array(x), np.array(y).ravel(), cv=cv, verbose=0)
        df_predict[name] = y_pred
        df_evaluation.loc[row_index, 'Model_Name'] = name
        for i in scoring:
            text = 'test_'+i
            df_evaluation.loc[row_index, i] = -1*scores[text].mean()
        row_index += 1
    df_evaluation.rename(columns = {'neg_mean_absolute_error': 'MAE', 'neg_median_absolute_error': 'MEAE',
                                    'neg_mean_squared_error': 'MSE', 'neg_root_mean_squared_error': 'RMSE'}, inplace = True)
    df_evaluation.sort_values(by=['MAE'], ascending=True, inplace=True)
    df_evaluation.reset_index(drop=True, inplace=True)
    return (df_evaluation, df_predict)
from sklearn.model_selection import cross_validate, cross_val_predict, KFold

kfolds = KFold(n_splits=5, shuffle=True, random_state=1)
scoring = ['neg_mean_absolute_error',
           'neg_median_absolute_error', 
           'neg_mean_squared_error', 
           'neg_root_mean_squared_error']
# cross validate
df_score, df_preds = storm_model(X_train_new, y_target, models, kfolds, scoring)
df_score.style.background_gradient(cmap='jet')
# Compare val with preds
df_preds['LogSalePrice'] = y_target['SalePrice']
df_teste = pd.DataFrame({'SalePrice': np.floor(np.expm1(df_preds['LogSalePrice'])),
                         'Preds': np.floor(np.expm1(df_preds['GradientBoosting']))})
df_teste['dif_val_pred'] = df_teste['Preds'] - df_teste['SalePrice']
df_teste.tail(3)
# StackingRegressor
best_models = [('GradientBoosting', grad),
               ('XGBoost', xgboost),
               ('LightGBM', lgbm)]
stack = StackingRegressor(estimators=best_models, final_estimator=huber_model)
scores = cross_validate(stack, np.array(X_train_new), np.array(y_target).ravel(), 
                        cv=kfolds, scoring=scoring, n_jobs=-1, verbose=0)
print(f"MAE score: {-1*scores['test_neg_mean_absolute_error'].mean()}")
y_pred = cross_val_predict(stack, np.array(X_train_new), np.array(y_target).ravel(), 
                           cv=kfolds, verbose=0, n_jobs=-1)
with plt.style.context('Solarize_Light2'):
    fig = plt.figure(figsize=(18,10))
    df_preds['Stacking'] = y_pred
    rows = 40
    g = df_preds.head(rows)
    for index, i in enumerate(df_preds.drop(columns='LogSalePrice').columns):
        plt.subplot(3,4,index+1)
        plt.scatter(g.index, g['LogSalePrice'], edgecolors='black')
        plt.plot(g.index, g[i], color='Red')
        plt.title(i, color='#073642', style='italic')
    fig.suptitle('Single predictors versus stacked predictors', y=1.06, fontsize=18, color='#002b36',  weight='bold')
    fig.tight_layout(pad=1.0)
from datetime import datetime

# Fitting the models on train data
print('=' * 20, 'START Fitting', '=' * 20)
print('=' * 55)
print(datetime.now(), '\n')

print(datetime.now(), 'ExtraTrees')
extratree_fit = extratree.fit(np.array(X_train_new), np.array(y_target).ravel())
print(datetime.now(), 'GradientBoosting')
grad_fit = grad.fit(np.array(X_train_new), np.array(y_target).ravel())
print(datetime.now(), 'HistGradientBoosting')
hist_fit = hist.fit(np.array(X_train_new), np.array(y_target).ravel())
print(datetime.now(), 'XGBoost')
xgboost_fit = xgboost.fit(np.array(X_train_new), np.array(y_target).ravel())
print(datetime.now(), 'LightGBM')
lgbm_fit = lgbm.fit(np.array(X_train_new), np.array(y_target).ravel())
print(datetime.now(), 'LassoCV')
lasso_fit = lasso.fit(np.array(X_train_new), np.array(y_target).ravel())
print(datetime.now(), 'ElasticNetCV')
elasticnet_fit = elasticnet.fit(np.array(X_train_new), np.array(y_target).ravel())
print(datetime.now(), 'Huber')
huber_fit = huber.fit(np.array(X_train_new), np.array(y_target).ravel())
print(datetime.now(), 'SVR')
svr_fit = svr.fit(np.array(X_train_new), np.array(y_target).ravel())
print(datetime.now(), 'StackingCVRegressor')
stack_fit = stack.fit(np.array(X_train_new), np.array(y_target).ravel())

print(datetime.now(), '\n')
print('=' * 20, 'FINISHED Fitting', '=' * 20)
print('=' * 58)
def blend_models_predict(X):
    return ((0.04* extratree_fit.predict(X)) + 
            (0.15 * grad_fit.predict(X)) + 
            (0.04 * hist_fit.predict(X)) + 
            (0.15 * xgboost_fit.predict(X)) + 
            (0.15 * lgbm_fit.predict(X)) + 
            (0.05 * lasso_fit.predict(X)) + 
            (0.05 * elasticnet_fit.predict(X)) + 
            (0.13 * huber_fit.predict(X)) +
            (0.04 * svr_fit.predict(X)) +
            (0.20 * stack_fit.predict(X)))
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission['SalePrice'] = np.floor(np.expm1(blend_models_predict(np.array(X_test_new))))
submission = submission[['Id', 'SalePrice']]
submission.head()
submission.to_csv('my_submission.csv', index=False)
print('Save submission', datetime.now())
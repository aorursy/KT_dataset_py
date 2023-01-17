# This block is from https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

import seaborn as sns #collection of functions for data visualization
print("seaborn version: {}". format(sns.__version__))

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder #OneHot Encoder

#misc libraries
import random
import time

%matplotlib inline

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

train_raw = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_raw = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample_raw = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train_raw.info()
def null_p (column):
    num_rows = train_raw.shape[0]
    null_percent = (column.isnull().sum() /num_rows)*100
    return null_percent
#drop columns with more than null_bar% of null data
def drop_null_col (df, null_bar):
    col_ls = []
    for col in df.columns:

        if null_p(df[col]) > null_bar:
            col_ls.append(col)
        
    return col_ls
#get a list of columns to drop
cols_drop = drop_null_col(train_raw, 40)
cols_drop
#generate a df with all the cols with 40% or more null data dropped
train_data_dnull = train_raw.copy().drop(cols_drop, axis = 1)
def null_p_df (df):
    p_ls = []
    type_ls = []
    for col in df.columns:
        n_percent = null_p(df[col])
        p_ls.append(n_percent)
        type_ls.append(df[col].dtype)
    np_df = pd.DataFrame({'Column Name': df.columns, "Null Percent": p_ls, "Data Type": type_ls})
    return np_df
np_df = null_p_df(train_data_dnull)
np_df.sort_values(by = 'Null Percent',  ascending=False).head(20)
#need special attention on LotFrontage, let's take a look at the distribution 
train_data_dnull.LotFrontage.hist()
#let's look at how it relates to our target
train_data_dnull.plot.scatter(x='SalePrice', y='LotFrontage', ylim=(0,400));
np_df.sort_values(by = 'Null Percent',  ascending=False).head(20)
from sklearn.impute import SimpleImputer

def clear_na (df):
    
    #let split the data for more targeted handling
    txt_cols = [cname for cname in df.columns if df[cname].dtype == "object"]

    # Select numerical columns
    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]

    
    txt_imputer = sklearn.impute.SimpleImputer(strategy='most_frequent')
    num_imputer = sklearn.impute.SimpleImputer(strategy='mean')
    
    txt_nonull = pd.DataFrame(txt_imputer.fit_transform(df[txt_cols]))
    txt_nonull.columns = txt_cols
    num_nonull = pd.DataFrame(num_imputer.fit_transform(df[num_cols]))
    num_nonull.columns = num_cols
    
    clean_df = txt_nonull.join(num_nonull)
    
    return clean_df
    
    
train_clean = clear_na(train_data_dnull)
train_clean.info()
#okay, now we have a clean dataset, let's take a look at the data more closely to understand what types of data we are dealing with
#to do this, we will break the data to txt and num, to allow easier handling

def txt_num_split(df):

    #let split the data for more targeted handling
    txt_cols = [cname for cname in df.columns if df[cname].dtype == "object"]
    # Select numerical columns
    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]
    
    txt_df = df[txt_cols]
    num_df = df[num_cols]
    
    return txt_df, num_df

train_clean_txt, train_clean_num = txt_num_split(train_clean)
train_clean_txt.describe()
uv_df = pd.DataFrame(train_clean_txt.describe().loc['unique'])
uv_df.sort_values(by = 'unique', ascending=False)
train_clean_txt.Neighborhood.value_counts()
#let's look at how it relates to our target
train_clean.plot.scatter(x='SalePrice', y='Neighborhood')

import category_encoders as ce

# Create the encoder
target_enc = ce.CatBoostEncoder(cols=['Neighborhood'])
target_enc_fitted = target_enc.fit(train_clean_txt['Neighborhood'], train_clean['SalePrice'])
nbh_trans = target_enc_fitted.transform(train_clean_txt['Neighborhood'])

nbh_trans
train_clean.Exterior2nd.value_counts()
train_clean.plot.scatter(x='SalePrice', y='Exterior2nd')
combine_ls = ['CBlock', 'Other', 'Stone', 'AsphShn', 'ImStucc', 'Brk Cmn']
train_clean_txt.Exterior2nd = train_clean_txt.Exterior2nd.apply(lambda x: 'Other' if x in combine_ls else x)
train_clean_txt.Exterior2nd.value_counts()
train_clean_txt.Exterior1st.value_counts()
combine_ls_e1 = ['CBlock', 'Other', 'Stone', 'AsphShn', 'ImStucc', 'BrkComm']
train_clean_txt.Exterior1st = train_clean_txt.Exterior1st.apply(lambda x: 'Other' if x in combine_ls_e1 else x)
train_clean_txt.Exterior1st.value_counts()
train_clean_num.hist(figsize = (20,20))
num_cat_cols = ['GarageYrBlt', 'MoSold', 'YrSold', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd']
def one_hot (raw_df, cat_cols):
    
    df = raw_df.copy()
    
    #next we apply One-Hot
    OH_en = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_source_df_imputed_col = OH_en.fit_transform(df[cat_cols])
    OH_source_df_imputed_col = pd.DataFrame(OH_source_df_imputed_col)
    
    #alining index 
    OH_source_df_imputed_col.index = df.index
    #alining columns
    OH_source_df_imputed_col.columns = OH_en.get_feature_names(cat_cols)
    return OH_source_df_imputed_col
onehot_test = one_hot(train_clean_num, num_cat_cols)
onehot_test
num_toscale = ['LotFrontage', 'LotArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'WoodDeckSF',
       'OpenPorchSF', 'PoolArea']
from mlxtend.preprocessing import minmax_scaling

def scaler (raw_df, cols):
    df = raw_df.copy()
    scaled_data = minmax_scaling(df , columns=cols)
    return scaled_data 
scaler_test = scaler(train_clean_num, num_toscale)
scaler_test
norm_ls = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'MSSubClass', 'OpenPorchSF', 'WoodDeckSF', 'SalePrice']
# for Box-Cox Transformation
from scipy import stats

def norm_trans (col_ls, target_df):
    df = target_df.copy()

    for col in df.columns:

        if col in col_ls:
            norm_data = stats.boxcox(df[col].array+0.001)
            norm_series = pd.Series(norm_data[0])
            norm_series.name = col
            df[col] = norm_series

    return df
train_clean_num_norm= norm_trans(norm_ls, train_clean_num)
train_clean_num_norm[norm_ls].hist(bins = 20, figsize = (20,20))
#a quick a dirty way to make sure we are not dealing with outliers

def replace_outliers (raw_df, cols):
    df = raw_df.copy()
    for col in cols:
        bounds = np.percentile(df[col], (0.1,99.9))
        low_bound = bounds[0]
        up_bound = bounds[1]
        df[col] = df[col].apply(lambda x: low_bound if x < low_bound else (up_bound if x > up_bound else x))
    
    return df
df_no_outliers = replace_outliers (train_clean_num_norm, norm_ls)
df_no_outliers[norm_ls].hist(bins = 20, figsize = (20,20))
def data_clean (raw_df):
    df = raw_df.copy()
    cols_to_drop = drop_null_col(df, 40)
    df = df.drop(cols_to_drop, axis = 1)
    df = clear_na(df)
    return df 
#creating combined dataframe for process

train_raw_lable = train_raw.copy()
train_raw_lable['IsTrain'] = 1
y_train = train_raw_lable.SalePrice.copy()
train_raw_lable = train_raw_lable.drop('SalePrice', axis = 1)

test_raw_lable = test_raw.copy()
test_raw_lable['IsTrain'] = 0
all_data_raw = pd.concat([train_raw_lable, test_raw_lable], ignore_index=True, sort=False)

all_data_raw
all_clean = data_clean (all_data_raw)
all_clean.loc[all_clean.IsTrain == 1]
all_clean.isnull().sum()
X_all_txt, X_all_num = txt_num_split(all_clean)
import category_encoders as ce

def ce_encoding (fit_df, trans_df, cols):
    df = trans_df.copy()
    
    target_enc = ce.CatBoostEncoder(cols=cols)
    target_enc_fitted = target_enc.fit(fit_df[cols], y_train)
    tran_cols = target_enc_fitted.transform(df[cols])
    return tran_cols
def txt_col_preprocess (raw_df):
    
    df = raw_df.copy()
    
    #aggregate scattered values in Exterior2nd 
    combine_ls_e2 = ['CBlock', 'Other', 'Stone', 'AsphShn', 'ImStucc', 'Brk Cmn']
    df.Exterior2nd = df.Exterior2nd.apply(lambda x: 'Other' if x in combine_ls_e2 else x)
    
    #aggregate scattered values in Exterior1st 
    combine_ls_e1 = ['CBlock', 'Other', 'Stone', 'AsphShn', 'ImStucc', 'BrkComm']
    df.Exterior1st = df.Exterior1st.apply(lambda x: 'Other' if x in combine_ls_e1 else x)
    
    #encode cols with more than 10 unique values with category_encoders
    top_uni_cols = ['Neighborhood', 'Exterior1st', 'Exterior2nd']
    ce_fit = all_clean.loc[all_clean.IsTrain == 1].copy()
    ce_cols = ce_encoding(ce_fit, df, top_uni_cols)
    ce_cols = scaler(ce_cols, top_uni_cols)
    df = df.drop(top_uni_cols, axis = 1).join(ce_cols)
    
    #onehot encode the rest
    rest_txt_cols = df.columns.drop(top_uni_cols)
    col_for_oh = one_hot(df, rest_txt_cols)
    df = df.drop(rest_txt_cols, axis = 1).join(col_for_oh)
    

    
    return df
    
    
    
def num_col_preprocess (raw_df):
    
    df = raw_df.copy()
    
    norm_ls = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'MSSubClass', 'OpenPorchSF', 'WoodDeckSF']
    num_toscale = ['LotFrontage', 'LotArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'WoodDeckSF',
       'OpenPorchSF', 'PoolArea']
    num_cat_cols = ['GarageYrBlt', 'MoSold', 'YrSold', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd']
    
    #create new features
    df['HasPorach'] = (df['OpenPorchSF'] + df['EnclosedPorch'])>0
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['HasPool'] = df['PoolArea'] > 0
    new_cols = ['HasPorach', 'HasPool']
    oh_new_cols = one_hot(df, new_cols)
    df = df.drop(new_cols, axis = 1).join(oh_new_cols)
    
    #normalize and onehot other cols
    df = replace_outliers (df, norm_ls)
    df = norm_trans (norm_ls, df)
    oh_trnfm_cols = one_hot(df, num_cat_cols)
    df = df.drop(num_cat_cols, axis = 1).join(oh_trnfm_cols)
    
    #scale cols
    scale_cols = scaler(df, num_toscale)
    df = df.drop(num_toscale, axis = 1).join(scale_cols)
    

    
    
    return df
    
X_all_txt_processed = txt_col_preprocess (X_all_txt)
X_all_txt_processed
X_all_num_processed = num_col_preprocess(X_all_num)
X_all_num_processed
#lastly, we will normalize SalePrice separately
norm_ls = stats.boxcox(y_train.array+0.001)
y_train_norm, norm_p = norm_ls[0], norm_ls[1]
y_train_norm
X_all_processed = X_all_txt_processed.join(X_all_num_processed)
X = X_all_processed.loc[X_all_processed['IsTrain']==1]
X = X.drop('IsTrain', axis = 1)
X_test = X_all_processed.loc[X_all_processed['IsTrain']==0]
X_test = X_test.drop('IsTrain', axis = 1)
# Loading neccesary packages for modelling.

from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor # This is for stacking part, works well with sklearn and others...
# Setting kfold for future use.

kf = KFold(10, random_state=42)
# Some parameters for ridge, lasso and elasticnet.

alphas_alt = [15.5, 15.6, 15.7, 15.8, 15.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [
    5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008
]
e_alphas = [
    0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007
]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

# ridge_cv

ridge = make_pipeline(RobustScaler(), RidgeCV(
    alphas=alphas_alt,
    cv=kf,
))

# lasso_cv:

lasso = make_pipeline(
    RobustScaler(),
    LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kf))

# elasticnet_cv:

elasticnet = make_pipeline(
    RobustScaler(),
    ElasticNetCV(max_iter=1e7,
                 alphas=e_alphas,
                 cv=kf,
                 random_state=42,
                 l1_ratio=e_l1ratio))

# svr:

svr = make_pipeline(RobustScaler(),
                    SVR(C=21, epsilon=0.0099, gamma=0.00017, tol=0.000121))

# gradientboosting:

gbr = GradientBoostingRegressor(n_estimators=2900,
                                learning_rate=0.0161,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=17,
                                loss='huber',
                                random_state=42)

# lightgbm:

lightgbm = LGBMRegressor(objective='regression',
                         n_estimators=3500,
                         num_leaves=5,
                         learning_rate=0.00721,
                         max_bin=163,
                         bagging_fraction=0.35711,
                         n_jobs=-1,
                         bagging_seed=42,
                         feature_fraction_seed=42,
                         bagging_freq=7,
                         feature_fraction=0.1294,
                         min_data_in_leaf=8)


# hist gradient boosting regressor:

hgrd= HistGradientBoostingRegressor(    loss= 'least_squares',
    max_depth= 2,
    min_samples_leaf= 40,
    max_leaf_nodes= 29,
    learning_rate= 0.15,
    max_iter= 225,
                                    random_state=42)


#to replace xgboost for meta_regressor
rf = RandomForestRegressor()


# stacking regressor:

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr,
                                         lightgbm,hgrd),
                                meta_regressor=rf,
                                use_features_in_secondary=True)
def model_check(X, y, estimators, cv):
    
    ''' A function for testing multiple estimators.'''
    
    model_table = pd.DataFrame()

    row_index = 0
    for est, label in zip(estimators, labels):

        MLA_name = label
        model_table.loc[row_index, 'Model Name'] = MLA_name

        cv_results = cross_validate(est,
                                    X,
                                    y,
                                    cv=cv,
                                    scoring='neg_root_mean_squared_error',
                                    return_train_score=True,
                                    n_jobs=-1)

        model_table.loc[row_index, 'Train RMSE'] = -cv_results[
            'train_score'].mean()
        model_table.loc[row_index, 'Test RMSE'] = -cv_results[
            'test_score'].mean()
        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        row_index += 1

    model_table.sort_values(by=['Test RMSE'],
                            ascending=True,
                            inplace=True)

    return model_table
# Setting list of estimators and labels for them:

estimators = [ridge, lasso, elasticnet, gbr, rf, lightgbm, svr, hgrd]
labels = [
    'Ridge', 'Lasso', 'Elasticnet', 'GradientBoostingRegressor',
    'RandomForestRegressor', 'LGBMRegressor', 'SVR', 'HistGradientBoostingRegressor',
]
# Executing cross validation.

raw_models = model_check(X, y_train_norm, estimators, kf)
raw_models
from datetime import datetime
# Fitting the models on train data.

print('=' * 20, 'START Fitting', '=' * 20)
print('=' * 55)

print(datetime.now(), 'StackingCVRegressor')
stack_gen_model = stack_gen.fit(X.values, y_train_norm)
print(datetime.now(), 'Elasticnet')
elastic_model_full_data = elasticnet.fit(X, y_train_norm)
print(datetime.now(), 'Lasso')
lasso_model_full_data = lasso.fit(X, y_train_norm)
print(datetime.now(), 'Ridge')
ridge_model_full_data = ridge.fit(X, y_train_norm)
print(datetime.now(), 'SVR')
svr_model_full_data = svr.fit(X, y_train_norm)
print(datetime.now(), 'GradientBoosting')
gbr_model_full_data = gbr.fit(X, y_train_norm)
print(datetime.now(), 'Hist')
hist_full_data = hgrd.fit(X, y_train_norm)

print('=' * 20, 'FINISHED Fitting', '=' * 20)
print('=' * 58)
def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) +
            (0.1 * lasso_model_full_data.predict(X)) +
            (0.1 * ridge_model_full_data.predict(X)) +
            (0.1 * svr_model_full_data.predict(X)) +
            (0.15 * gbr_model_full_data.predict(X)) +
            (0.1 * hist_full_data.predict(X)) +
            (0.35 * stack_gen_model.predict(X.values)))
y_sub_blend_raw = blend_models_predict(X_test)
from scipy.special import inv_boxcox
y_sub_blend = inv_boxcox(y_sub_blend_raw, norm_p)
y_sub_gbr_raw = gbr_model_full_data.predict(X_test)
y_sub_gbr = inv_boxcox(y_sub_gbr_raw, norm_p)
sub_blend = pd.DataFrame({'ID': sample_raw['Id'], 'SalePrice': y_sub_blend})
sub_blend.to_csv('sub_blend.csv', index = False)
sub_gbr = pd.DataFrame({'ID': sample_raw['Id'], 'SalePrice': y_sub_gbr})
sub_gbr.to_csv('sub_gbr.csv', index = False)
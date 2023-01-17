import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, Lasso
import numpy as np
import seaborn as sns
from scipy.stats import norm 
from scipy import stats
import matplotlib.pyplot as plt
from math import ceil
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer
import time
from mlxtend.regressor import StackingCVRegressor
df_csv_train = pd.read_csv('../input/train.csv', index_col='Id')
df_csv_test = pd.read_csv('../input/test.csv', index_col='Id')
print('csv train data shape: ', df_csv_train.shape)
print('csv test data shape: ', df_csv_test.shape)
df_all = pd.concat([df_csv_train, df_csv_test], sort=False)
df_all.head()
def get_na_cols():
    cols_with_na = df_all.drop('SalePrice', axis=1).isnull().sum()
    cols_with_na = cols_with_na[cols_with_na > 0]
    return cols_with_na
print('NA columns:\n', get_na_cols().sort_values(ascending=False).to_string())
# columns where NA values have meaning, e.g. no pool, no basement, etc.
cols_fillna = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu',
               'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType',
               'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']

# replace 'NA' with 'None' in these columns
for col in cols_fillna:
    df_all[col].fillna('None', inplace=True)
# Fill with YearBuilt
df_all.loc[df_all.GarageYrBlt.isnull(), 'GarageYrBlt'] = df_all.loc[df_all.GarageYrBlt.isnull(), 'YearBuilt']
# No masonry veneer - fill area with 0
df_all.MasVnrArea.fillna(0, inplace=True)

# No basement - fill areas/counts with 0
df_all.BsmtFullBath.fillna(0, inplace=True)
df_all.BsmtHalfBath.fillna(0, inplace=True)
df_all.BsmtFinSF1.fillna(0, inplace=True)
df_all.BsmtFinSF2.fillna(0, inplace=True)
df_all.BsmtUnfSF.fillna(0, inplace=True)
df_all.TotalBsmtSF.fillna(0, inplace=True)

# No garage - fill areas/counts with 0
df_all.GarageArea.fillna(0, inplace=True)
df_all.GarageCars.fillna(0, inplace=True)
# function to normalise a column of values to lie between 0 and 1
def scale_minmax(col):
    return (col - col.min()) / (col.max() - col.min())
# convert categoricals to dummies, exclude SalePrice from model
df_frontage = pd.get_dummies(df_all.drop('SalePrice', axis=1))

# normalise columns to 0-1
for col in df_frontage.drop('LotFrontage', axis=1).columns:
    df_frontage[col] = scale_minmax(df_frontage[col])

lf_train = df_frontage.dropna()
lf_train_y = lf_train.LotFrontage
lf_train_X = lf_train.drop('LotFrontage', axis=1)

# fit model
lr = Ridge()
lr.fit(lf_train_X, lf_train_y)

# check model results
lr_coefs = pd.Series(lr.coef_, index=lf_train_X.columns)

print('----------------')
print('Intercept:', lr.intercept_)
print('----------------coefficient: head(10)')
print(lr_coefs.sort_values(ascending=False).head(10))
print('----------------coefficient: tail(10)')
print(lr_coefs.sort_values(ascending=False).tail(10))
print('----------------')
print('R2:', lr.score(lf_train_X, lf_train_y))
print('----------------')

# fill na values using model predictions
na_frontage = df_all.LotFrontage.isnull()
X = df_frontage[na_frontage].drop('LotFrontage', axis=1)
y = lr.predict(X)

# fill na values
df_all.loc[na_frontage, 'LotFrontage'] = y
# fill remaining NA with mode in that column
for col in get_na_cols().index:
    df_all[col].fillna(df_all[col].mode()[0], inplace=True)
print('NA columns:\n', get_na_cols().sort_values(ascending=False).to_string())
# create separate columns for area of each possible
# basement finish type
bsmt_fin_cols = ['BsmtGLQ', 'BsmtALQ', 'BsmtBLQ',
                 'BsmtRec', 'BsmtLwQ']

for col in bsmt_fin_cols:
    # initialise as columns of zeros
    df_all[col + 'SF'] = 0

# fill remaining finish type columns
for row in df_all.index:
    fin1 = df_all.loc[row, 'BsmtFinType1']
    if (fin1 != 'None') and (fin1 != 'Unf'):
        # add area (SF) to appropriate column
        df_all.loc[row, 'Bsmt' + fin1 + 'SF'] += df_all.loc[row, 'BsmtFinSF1']

    fin2 = df_all.loc[row, 'BsmtFinType2']
    if (fin2 != 'None') and (fin2 != 'Unf'):
        df_all.loc[row, 'Bsmt' + fin2 + 'SF'] += df_all.loc[row, 'BsmtFinSF2']

# remove initial BsmtFin columns
df_all.drop(['BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2'], axis=1, inplace=True)

# already have BsmtUnf column in dataset
bsmt_fin_cols.append('BsmtUnf')

# also create features representing the fraction of the basement that is each finish type
for col in bsmt_fin_cols:
    df_all[col + 'Frac'] = df_all[col + 'SF'] / df_all['TotalBsmtSF']
    # replace any NA with zero (for properties without a basement)
    df_all[col + 'Frac'].fillna(0, inplace=True)
df_all['LowQualFinFrac'] = df_all['LowQualFinSF'] / df_all['GrLivArea']
df_all['1stFlrFrac'] = df_all['1stFlrSF'] / df_all['GrLivArea']
df_all['2ndFlrFrac'] = df_all['2ndFlrSF'] / df_all['GrLivArea']
df_all['TotalAreaSF'] = df_all['GrLivArea'] + df_all['TotalBsmtSF'] + df_all['GarageArea'] + df_all['EnclosedPorch'] + \
                        df_all['ScreenPorch']
df_all['LivingAreaSF'] = df_all['1stFlrSF'] + df_all['2ndFlrSF'] + df_all['BsmtGLQSF'] + df_all['BsmtALQSF'] + \
                         df_all['BsmtBLQSF']
df_all['StorageAreaSF'] = df_all['LowQualFinSF'] + df_all['BsmtRecSF'] + df_all['BsmtLwQSF'] + df_all['BsmtUnfSF'] + \
                          df_all['GarageArea']
cols_ExGd = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
             'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
             'GarageCond', 'PoolQC']
dict_ExGd = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
for col in cols_ExGd:
    df_all[col].replace(dict_ExGd, inplace=True)
print(df_all[cols_ExGd].head())
# Remaining columns
df_all['BsmtExposure'].replace({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}, inplace=True)
df_all['CentralAir'].replace({'Y': 1, 'N': 0}, inplace=True)
df_all['Functional'].replace(
    {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0},
    inplace=True)
df_all['GarageFinish'].replace({'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}, inplace=True)
df_all['LotShape'].replace({'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0}, inplace=True)
df_all['Utilities'].replace({'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0}, inplace=True)
df_all['LandSlope'].replace({'Gtl': 2, 'Mod': 1, 'Sev': 0}, inplace=True)
# fraction of zeros in each column
frac_zeros = ((df_all == 0).sum() / len(df_all))

# no. unique values in each column
n_unique = df_all.nunique()

# difference between frac. zeros and expected
# frac. zeros if values evenly distributed between
# classes
xs_zeros = frac_zeros - 1 / n_unique

# create dataframe and display which columns may be problematic
zero_cols = pd.DataFrame({'frac_zeros': frac_zeros, 'n_unique': n_unique, 'xs_zeros': xs_zeros})
zero_cols = zero_cols[zero_cols.frac_zeros > 0]
zero_cols.sort_values(by='xs_zeros', ascending=False, inplace=True)
print(zero_cols[(zero_cols.xs_zeros > 0)])
# very few properties with Pool or 3SsnPorch
# replace columns with binary indicator
df_all['HasPool'] = (df_all['PoolQC'] > 0).astype(int)
df_all['Has3SsnPorch'] = (df_all['3SsnPorch'] > 0).astype(int)
df_all.drop(['PoolQC', 'PoolArea', '3SsnPorch'], axis=1, inplace=True)
# 'half' bathrooms - add half value to 'full' bathrooms
df_all['BsmtFullBath'] = df_all['BsmtFullBath'] + 0.5 * df_all['BsmtHalfBath']
df_all['FullBath'] = df_all['FullBath'] + 0.5 * df_all['HalfBath']
df_all.drop(['BsmtHalfBath', 'HalfBath'], axis=1, inplace=True)    
# create additional dummy variable for
# continuous variables with a lot of zeros
dummy_cols = ['LowQualFinSF', '2ndFlrSF',
              'MiscVal', 'ScreenPorch', 'WoodDeckSF', 'OpenPorchSF',
              'EnclosedPorch', 'MasVnrArea', 'GarageArea', 'Fireplaces',
              'BsmtGLQSF', 'BsmtALQSF', 'BsmtBLQSF', 'BsmtRecSF',
              'BsmtLwQSF', 'BsmtUnfSF', 'TotalBsmtSF']
for col in dummy_cols:
    df_all['Has' + col] = (df_all[col] > 0).astype(int)
df_saleprice = df_all.SalePrice.dropna()
df_all.SalePrice = np.log(df_all.SalePrice)
print("SalePrice Skewness: %f" % df_all.SalePrice.skew())
print("SalePrice Kurtosis: %f" % df_all.SalePrice.kurt())
sns.distplot(df_saleprice, fit=norm)
fig = plt.figure()
res = stats.probplot(df_all.SalePrice, plot=plt)
# extract names of numeric columns
dtypes = df_all.dtypes
cols_numeric = dtypes[dtypes != object].index.tolist()

# MSubClass should be treated as categorical
cols_numeric.remove('MSSubClass')

# choose any numeric column with less than 13 values to be
# "discrete". 13 chosen to include months of the year.
# other columns "continuous"
col_nunique = dict()
for col in cols_numeric:
    col_nunique[col] = df_all[col].nunique()
col_nunique = pd.Series(col_nunique)
cols_discrete = col_nunique[col_nunique < 13].index.tolist()
cols_continuous = col_nunique[col_nunique >= 13].index.tolist()

print(len(cols_numeric), 'numeric columns, of which',
              len(cols_continuous), 'are continuous and',
              len(cols_discrete), 'are discrete.')

# extract names of categorical columns
cols_categ = dtypes[~dtypes.index.isin(cols_numeric)].index.tolist()
for col in cols_categ:
    df_all[col] = df_all[col].astype('category')

print(len(cols_categ), 'categorical columns.')
cols_categ_first6 = cols_categ[:6]
fcols = 3
frows = ceil(len(cols_categ_first6)/fcols)
plt.figure(figsize=(15,4*frows))

for i,col in enumerate(cols_categ_first6):
    plt.subplot(frows,fcols,i+1)
    sns.violinplot(df_all[col],df_all['SalePrice'])
# anova test to check significance of variation in column 'group' vs. column 'value' 
def anova(group,value):
    # select columns of interest, and remove any rows with nan values
    data = df_all[[group,value]]
    data = data[~(data[group].isnull() | data[value].isnull())]
    
    # stats across all data
    tot_groups = data[group].nunique() # no. of groups
    len_data = len(data) # total sample size of houses (all groups)
    mean_data = data[value].mean() # mean across all groups
    df_betwn = tot_groups - 1 # degrees of freedom betwn grps
    df_within = len_data - tot_groups # degrees of freedom within grps
    
    # per group stats
    n_in_group = data.groupby(group)[value].count() # no. houses in group
    mean_group = data.groupby(group)[value].mean() # mean value in this group
    
    # between-group variability
    betwn_var = n_in_group*((mean_group - mean_data)**2)
    betwn_var = float(betwn_var.sum())/df_betwn
    
    # within-group variability
    within_var = 0
    for grp in data[group].unique():
        samples = data.loc[data[group]==grp, value]
        within_var += ((samples-mean_group[grp])**2).sum()
        
    within_var = float(within_var)/df_within
    
    #F-test statistic
    F = betwn_var/within_var
    
    # p-value
    p = stats.f.sf(F, df_betwn, df_within)
    
    return p      
# check significance of categorical variables on SalePrice
p_col = dict()
for col in cols_categ:
    p_col[col] = anova(col,'SalePrice')
pd.Series(p_col).sort_values(ascending=False).head()
cols_discrete_first6 = cols_discrete[:6]
fcols = 3
frows = ceil(len(cols_discrete_first6)/fcols)
plt.figure(figsize=(15,4*frows))

for i,col in enumerate(cols_discrete_first6):
    plt.subplot(frows,fcols,i+1)
    sns.violinplot(df_all[col],df_all['SalePrice'])
p_col = dict()
for col in cols_discrete:
    p_col[col] = anova(col,'SalePrice')
pd.Series(p_col).sort_values(ascending=False).head()
cols_continuous_first3 = cols_continuous[:3]
fcols = 2
frows = len(cols_continuous_first3)
plt.figure(figsize=(5*fcols,4*frows))

i=0
for col in cols_continuous_first3:
    i+=1
    ax=plt.subplot(frows,fcols,i)
    sns.regplot(x=col, y='SalePrice', data=df_all, ax=ax, 
                scatter_kws={'marker':'.','s':3,'alpha':0.3},
                line_kws={'color':'k'});
    plt.xlabel(col)
    plt.ylabel('SalePrice')
    
    i+=1
    ax=plt.subplot(frows,fcols,i)
    sns.distplot(df_all[col].dropna() , fit=stats.norm)
    plt.xlabel(col)
df_corr = df_all.loc[df_csv_train.index, cols_numeric].corr(method='spearman').abs()
# order columns and rows by correlation with SalePrice
df_corr = df_corr.sort_values('SalePrice', axis=0, ascending=False).sort_values('SalePrice', axis=1,
                                                                                ascending=False)

print(df_corr.SalePrice.head())
print('-----------------')
print(df_corr.SalePrice.tail())

ax=plt.figure(figsize=(20,16)).gca()
sns.heatmap(df_corr,ax=ax,square=True)
sns.regplot(x='GarageCars',y='GarageArea',data=df_all)
scale_cols = [col for col in cols_numeric if col != 'SalePrice']
df_all[scale_cols] = df_all[scale_cols].apply(scale_minmax, axis=0)
df_all[scale_cols].describe()
cols_continuous_first3 = cols_continuous[:3]
fcols = 6
frows = len(cols_continuous_first3)
plt.figure(figsize=(4*fcols,4*frows))
i=0

for var in cols_continuous_first3:
    if var!='SalePrice':
        dat = df_all[[var, 'SalePrice']].dropna()
        
        i+=1
        plt.subplot(frows,fcols,i)
        sns.distplot(dat[var] , fit=stats.norm);
        plt.title(var+' Original')
        plt.xlabel('')
        
        i+=1
        plt.subplot(frows,fcols,i)
        _=stats.probplot(dat[var], plot=plt)
        plt.title('skew='+'{:.4f}'.format(stats.skew(dat[var])))
        plt.xlabel('')
        plt.ylabel('')
        
        i+=1
        plt.subplot(frows,fcols,i)
        plt.plot(dat[var], dat['SalePrice'],'.',alpha=0.5)
        plt.title('corr='+'{:.2f}'.format(np.corrcoef(dat[var], dat['SalePrice'])[0][1]))
 
        i+=1
        plt.subplot(frows,fcols,i)
        trans_var, lambda_var = stats.boxcox(dat[var].dropna()+1)
        trans_var = scale_minmax(trans_var)      
        sns.distplot(trans_var , fit=stats.norm);
        plt.title(var+' Tramsformed')
        plt.xlabel('')
        
        i+=1
        plt.subplot(frows,fcols,i)
        _=stats.probplot(trans_var, plot=plt)
        plt.title('skew='+'{:.4f}'.format(stats.skew(trans_var)))
        plt.xlabel('')
        plt.ylabel('')
        
        i+=1
        plt.subplot(frows,fcols,i)
        plt.plot(trans_var, dat['SalePrice'],'.',alpha=0.5)
        plt.title('corr='+'{:.2f}'.format(np.corrcoef(trans_var,dat['SalePrice'])[0][1]))
# variables not suitable for box-cox transformation (usually due to excessive zeros)
cols_notransform = ['2ndFlrSF', '1stFlrFrac', '2ndFlrFrac', 'StorageAreaSF',
                    'EnclosedPorch', 'LowQualFinSF', 'MasVnrArea',
                    'MiscVal', 'ScreenPorch', 'OpenPorchSF', 'WoodDeckSF', 'SalePrice',
                    'BsmtGLQSF', 'BsmtALQSF', 'BsmtBLQSF', 'BsmtRecSF', 'BsmtLwQSF', 'BsmtUnfSF',
                    'BsmtGLQFrac', 'BsmtALQFrac', 'BsmtBLQFrac', 'BsmtRecFrac', 'BsmtLwQFrac', 'BsmtUnfFrac']
cols_transform = [col for col in cols_continuous if col not in cols_notransform]

# transform remaining variables
print('Transforming', len(cols_transform), 'columns:', cols_transform)

for col in cols_transform:
    # transform column
    df_all.loc[:, col], _ = stats.boxcox(df_all.loc[:, col] + 1)
    # renormalise column
    df_all.loc[:, col] = scale_minmax(df_all.loc[:, col])
# select which features to use
model_cols = df_all.columns

# encode categoricals
df_model = pd.get_dummies(df_all[model_cols])

# Rather than including Condition1 and Condition2, or Exterior1st and Exterior2nd,
# combine the dummy variables (allowing 2 true values per property)
if ('Condition1' in model_cols) and ('Condition2' in model_cols):
    cond_suffix = ['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNn']
    for suffix in cond_suffix:
        col_cond1 = 'Condition1_' + suffix
        col_cond2 = 'Condition2_' + suffix

        df_model[col_cond1] = df_model[col_cond1] | df_model[col_cond2]
        df_model.drop(col_cond2, axis=1, inplace=True)

if ('Exterior1st' in model_cols) and ('Exterior2nd' in model_cols):
    # some different strings in Exterior1st and Exterior2nd for same type - rename columns to correct
    df_model.rename(columns={'Exterior2nd_Wd Shng': 'Exterior2nd_WdShing',
                                  'Exterior2nd_Brk Cmn': 'Exterior2nd_BrkComm',
                                  'Exterior2nd_CmentBd': 'Exterior2nd_CemntBd'}, inplace=True)
    ext_suffix = ['AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd',
                  'HdBoard', 'ImStucc', 'MetalSd', 'Plywood', 'Stone',
                  'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing', 'AsbShng']
    for suffix in ext_suffix:
        col_cond1 = 'Exterior1st_' + suffix
        col_cond2 = 'Exterior2nd_' + suffix
        df_model[col_cond1] = df_model[col_cond1] | df_model[col_cond2]
        df_model.drop(col_cond2, axis=1, inplace=True)
def get_rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff ** 2)
    n = len(y_pred)

    return np.sqrt(sum_sq / n)
# find and remove outliers using a Ridge model
def find_outliers(model, X, y, sigma=3):
    # predict y values using model
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # if predicting fails, try fitting the model first
    except:
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    # calculate residuals between the model prediction and true y values
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # calculate z statistic, define outliers to be where |z|>sigma
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    # print and plot the results
    print('R2=', model.score(X, y))
    print('rmse=', get_rmse(y, y_pred))
    print('---------------------------------------')

    print('mean of residuals:', mean_resid)
    print('std of residuals:', std_resid)
    print('---------------------------------------')

    print(len(outliers), 'outliers:')
    print(outliers.tolist())

    return outliers

df_model_train = df_model[~df_model.SalePrice.isnull()]
print('Before drop outliers, train data shape: ', df_model_train.shape)
outliers = find_outliers(Ridge(), df_model_train.drop('SalePrice', axis=1), df_model_train.SalePrice)
df_model_train = df_model_train.drop(outliers)
print('After drop outliers, train data shape: ', df_model_train.shape)
x_train = df_model_train.drop('SalePrice', axis=1)
y_train = df_model_train.SalePrice
# Ridge 
model_ridge = {"model_name": 'Ridge', 'model': Ridge(), 'param_grid': {'alpha': (np.arange(0.25, 6, 0.25))}}

# ElasticNet
model_elastic = {"model_name": 'ElasticNet', 'model': ElasticNet(),
                    'param_grid': {'alpha': np.arange(1e-4, 1e-3, 1e-4),
                                   'l1_ratio': np.arange(0.1, 1.0, 0.1),
                                   'max_iter': [100000]}
                    }

# GradientBoostingRegressor
model_gbr = {"model_name": 'GradientBoostingRegressor', 'model': GradientBoostingRegressor(),
                                   'param_grid': {'n_estimators': [150, 250, 350],
                                                  'max_depth': [1, 2, 3],
                                                  'min_samples_split': [5, 6, 7]}}
# SVR
model_svr = {"model_name": 'SVR', 'model': SVR(), 'param_grid': {'C': np.arange(1, 21, 2),
                                                                 'kernel': ['poly', 'rbf', 'sigmoid'],
                                                                 'gamma': ['auto']}}
# RandomForestRegressor
param_grid_rfr = {'n_estimators': [100, 150, 200],
                  'max_features': [25, 50, 75],
                  'min_samples_split': [2, 4, 6]}
model_rfr = {"model_name": 'RandomForestRegressor', 'model': RandomForestRegressor(),
                               'param_grid': param_grid_rfr}

# StackingCVRegressor
stregr = StackingCVRegressor(regressors=[Ridge(), Lasso()],
                             meta_regressor=ElasticNet(),
                             use_features_in_secondary=True)
param_alpha = [0.1, 1, 10]
elastic_net_param_alpha = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
elastic_net_param_l1_ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
param_grid_stack = {'ridge__alpha': param_alpha, 'lasso__alpha': param_alpha,
              'meta-elasticnet__alpha': elastic_net_param_alpha,
              'meta-elasticnet__l1_ratio': elastic_net_param_l1_ratio,
              }
model_stack = {"model_name": 'StackingCVRegressor', 'model': stregr,
                               'param_grid': param_grid_stack}

models = [model_ridge, model_elastic, model_gbr, model_svr, model_rfr]
# Add StackingCVRegressor to train model on my local PC
# models = [model_ridge, model_elastic, model_gbr, model_svr, model_rfr, model_stack]
def train_model(best_model, param_grid=[], X=[], y=[],
                splits=5, repeats=5):
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    gs = GridSearchCV(best_model, param_grid, cv=rkfold,
                      scoring=make_scorer(get_rmse, greater_is_better=False),
                      verbose=0, return_train_score=True)
    gs.fit(X.values, y.values)

    best_model = gs.best_estimator_
    best_idx = gs.best_index_
    grid_results = pd.DataFrame(gs.cv_results_)
    cv_mean = abs(grid_results.loc[best_idx, 'mean_test_score'])
    cv_std = grid_results.loc[best_idx, 'std_test_score']
    y_pred = best_model.predict(X)

    rsquare = '%.5f' % best_model.score(X, y)
    rmse = '%.5f' % get_rmse(y, y_pred)
    cv_mean = '%.5f' % (cv_mean)
    cv_std = '%.5f' % (cv_std)

    return best_model, (rmse, rsquare, cv_mean, cv_std), grid_results
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('once', category=ConvergenceWarning, append=True)
RANDOM_SEED = 33
np.random.seed(RANDOM_SEED)

# places to store optimal models and scores
best_model_instances = dict()
train_scores_on_models = pd.DataFrame(columns=['rmse', 'rsquare', 'cv_mean', 'cv_std', 'train_time'])

for model in models:
    model_name = model['model_name']
    model_instance = model['model']
    model_param_grid = model['param_grid']
    start = time.time()
    print(model_name, ' is training model...')
    best_model_instances[model_name], (rmse, rsquare, cv_mean, cv_std), grid_results = train_model(
        model_instance,
        param_grid=model_param_grid, X=x_train,
        y=y_train)
    end = time.time()
    train_time = '%.2fmin' % ((end - start) / 60)
    score = pd.Series(
        {'rmse': rmse, 'rsquare': rsquare, 'cv_mean': cv_mean, 'cv_std': cv_std, 'train_time': train_time})
    score.name = model_name
    train_scores_on_models = train_scores_on_models.append(score)
print('============= Final train score =============\n', train_scores_on_models.sort_values(by='rmse'),
      '\n=============================================')    
df_model_test = df_model[df_model.SalePrice.isnull()]
x_test = df_model_test.drop('SalePrice', axis=1)
y_test = df_model_test.SalePrice
pd.DataFrame({"Id": np.arange(1461, 2920), "SalePrice": np.exp(best_model_instances['Ridge'].predict(x_test))}).to_csv(
    'submission.csv', header=True, index=False)

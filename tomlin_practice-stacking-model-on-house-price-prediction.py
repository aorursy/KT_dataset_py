import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

from scipy import stats
from scipy.stats import norm, skew

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

# show all columns
pd.set_option('display.max_columns', 500)

# suppress warnings
import warnings 
warnings.simplefilter('ignore')


from subprocess import check_output
print(check_output(["ls"]).decode("utf8"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# remove unnecessary column
train_id = train['Id']
test_id = test['Id']
train.drop('Id', axis = 1, inplace=True)
test.drop('Id', axis = 1, inplace=True)

ntrain = train.shape[0]
ntest = test.shape[1]
print('train shape:', train.shape)
print('test shape:', test.shape)

SalePrice = train['SalePrice'].values

# create dataset flag
train['DataSet'] = 'train'
test['DataSet'] = 'test'

# 'SalePrice' will be excluded because of inner concatenation
all_data = pd.concat([train, test], ignore_index=True, join='inner')
print('Is column SalePrice in the dataset:' ,'SalePrice' in all_data.columns)

print('all data shape:', all_data.shape)
all_data.head(n=2)
import missingno as msno
msno.matrix(all_data, figsize=(25,5)) # show up the distribution of missing values in each row
#msno.bar(all_data, figsize=(25,5)) # show up the percentage of missing values in each column
# calculate the frequency of missing value

from operator import itemgetter

missing = (all_data.isnull().sum()/all_data.shape[0])
missing = zip(missing.index.values, missing)

# zip() in conjunction with the * operator can be used to unzip a list
ind, missing = zip(*sorted(missing, key=itemgetter(1), reverse=True))
missing = pd.DataFrame(np.array(missing), index=ind, columns=["MissingRate"])
missing.head(n=10)
# drop out feature of missing rate above 90%
drop_col = missing.loc[missing['MissingRate'] > 0.9].index.values
all_data_sub = all_data.drop(columns=drop_col)
# Through observation, the column name ends with `Qul` or `Cond` are the ones needed to be re-code into ordinal value

col_set1 = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
       'KitchenQual','FireplaceQu','GarageQual','GarageCond']

col_set2 = ['BsmtExposure']

col_set3 = ['BsmtFinType1','BsmtFinType2']

col_set4 = ['GarageFinish']

col_set5 = ['Fence']


# create a mapper for value replacement

mapper_set1 = {'NA':0,
               'Po':1,
               'Fa':2,
               'TA':3,
               'Gd':4,
               'Ex':5}

mapper_set2 = {'NA':0,
               'No':1,
               'Mn':2,
               'Av':3,
               'Gd':4}

mapper_set3 = {'NA':0,
               'Unf':1,
               'LwQ':2,
               'Rec':3,
               'BLQ':4,
               'ALQ':5,
               'GLQ':6}

mapper_set4 = {'NA':0,
               'Unf':1,
               'RFn':2,
               'Fin':3}

mapper_set5 = {'NA':0,
               'MnWw':1,
               'GdWo':2,
               'MnPrv':3,
               'GdPrv':4}

# when we read in the dataset, the NA value is mis-interpreted as null value
def replaceNa(df,col):
    for x in col:
        print('column {} null value is replaced to NA:'.format(x), df[x].isnull().any())
        df[x].fillna('NA', inplace=True)


replaceNa(all_data_sub, col_set1)
replaceNa(all_data_sub, col_set2)
replaceNa(all_data_sub, col_set3)
replaceNa(all_data_sub, col_set4)
replaceNa(all_data_sub, col_set5)

def printUniqueValue(df, col):
    for x in col:
        print('unique value of column {}:'.format(x), df[x].unique())

printUniqueValue(all_data_sub, col_set1)
printUniqueValue(all_data_sub, col_set2)
printUniqueValue(all_data_sub, col_set3)
printUniqueValue(all_data_sub, col_set4)
printUniqueValue(all_data_sub, col_set5)

def replaceToOrdinalNum(df, col, mapper):
    for x in col:
        df[x].replace(to_replace=mapper, inplace=True)
        print('column {} new replaced value:'.format(x), df[x].unique())

replaceToOrdinalNum(all_data_sub, col_set1, mapper_set1)
replaceToOrdinalNum(all_data_sub, col_set2, mapper_set2)
replaceToOrdinalNum(all_data_sub, col_set3, mapper_set3)
replaceToOrdinalNum(all_data_sub, col_set4, mapper_set4)
replaceToOrdinalNum(all_data_sub, col_set5, mapper_set5)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

def completeNumCol(df):
    '''return no-na numerical column names'''
    col = []
    # filter numerical column
    for x in df.columns.values:
        if df[x].dtype in (float,int):
            col.append(x)
    
    # filter no na col
    no_nan = df[col].isnull().any() == False
    col = list(no_nan.index.values[no_nan == True])
    return col

def noncompleteNumCol(df):
    '''return exist-na numerical column names'''
    col = []
    # filter numerical column
    for x in df.columns.values:
        if df[x].dtype in (float,int):
            col.append(x)
    
    # filter has na col
    has_nan = df[col].isnull().any() == True
    col = list(has_nan.index.values[has_nan == True])
    return col

def categoricalCol(df):
    '''return categorical column names'''
    # impute all categorical feature
    cat_col = []
    for x in df.columns.values:
        if df[x].dtype == object:
            cat_col.append(x)
    return cat_col


def imputeNumCol(df, num_col, fit_col):
    '''use no-na numerical column to calculate the distance among data points
        and impute the numerical column existing NA'''
    df = df.copy()
    for i in num_col:
        fit_col.append(i) # include y to check up for na
        knn_train = df[fit_col].dropna() # ensure no na value in train dataset
    
        knn_train_y = knn_train[i]
    
        std_scaler = StandardScaler()
        knn_train_x = knn_train.drop(columns=i) # drop y col
        std_scaler.fit(knn_train_x) # standardize for knn
        std_knn_train_x = std_scaler.transform(knn_train_x)
    
        neigh = KNeighborsRegressor(n_neighbors=7)
        neigh.fit(std_knn_train_x,knn_train_y)

        na_ind = df[df[i].isnull()].index.values # identify na index for y col
        fit_col.remove(i) # remove y col
        
        test_x = df[fit_col] # for knn prediction
        std_test_x = std_scaler.transform(test_x) 
        
        if not len(na_ind) == 0:
            value = neigh.predict(test_x.iloc[na_ind]) # predict y col on na rows
            df[i][na_ind]= value
        else:
            continue

    return df


def imputeCatCol(df, cat_col, fit_col):
    '''use no-na numerical column to calculate the distance among data points
        and impute the categorical column existing NA'''    
    df = df.copy()
    for i in cat_col:
        fit_col.append(i) # include y to check up for na
        knn_train = df[fit_col].dropna() # ensure no na value in train dataset
    
        knn_train_y = knn_train[i]
    
        std_scaler = StandardScaler()
        knn_train_x = knn_train.drop(columns=i) # drop y col
        std_scaler.fit(knn_train_x) # standardize for knn
        std_knn_train_x = std_scaler.transform(knn_train_x)
    
        neigh = KNeighborsClassifier(n_neighbors=7)
        neigh.fit(std_knn_train_x,knn_train_y)

        na_ind = df[df[i].isnull()].index.values # identify na index for y col
        fit_col.remove(i) # remove y col
        
        test_x = df[fit_col] # for knn prediction
        std_test_x = std_scaler.transform(test_x) 
        
        if not len(na_ind) == 0:
            value = neigh.predict(test_x.iloc[na_ind]) # predict y col on na rows
            df[i][na_ind]= value
        else:
            continue

    return df




def imputeMain(df):
    '''define the imputation workflow'''
    fit_col = completeNumCol(df)
    num_col = noncompleteNumCol(df)
    cat_col = categoricalCol(df)
    
    df = imputeNumCol(df, num_col, fit_col)
    df = imputeCatCol(df, cat_col, fit_col)
    
    return df

all_data_sub_impute = imputeMain(all_data_sub)
all_data_sub.isnull().any().sum()
# check again if all missing values are imputed. 
all_data_sub_impute.isnull().any().sum()
# get the parameters of normal distribution based on the data 'SalePrice'
(mu, sigma) = norm.fit(SalePrice)
print('SalePrice mu:', mu)
print('SalePrice sigma:', sigma)
# using option `fit = norm` to compare the data to the theoretical distribution
from scipy.stats import norm

fig, ax = plt.subplots(1,2, figsize=(10,5))
stats.probplot(train['SalePrice'], plot=ax[0], sparams=(mu, sigma))
sns.distplot(SalePrice, kde=True ,fit=norm, ax=ax[1], axlabel='SalePrice') 
ax[0].set_title('') # remove plot title
def numCol(df, data_type=[]):
    '''return numerical column names for distribution visualization'''
    num_col = []
    for x in df.columns.values:
        if df[x].dtype in data_type:
            num_col.append(x)
    return num_col

num_col = numCol(all_data_sub_impute, data_type=[int,float])
print('number of num_col:', len(num_col))
print(num_col)
import math

def printChart(df, num_col):
    '''print the distribution of num_col from df'''
    fig, ax = plt.subplots(math.ceil(len(num_col)/2),4, figsize=(20,80))
    ax = ax.flatten()
    print('ax len:',len(ax))
    
    index = 0
    for x in num_col:
        (mu, sigma) = norm.fit(df[x])
        stats.probplot(df[x],plot=ax[index],sparams=(mu,sigma))
        ax[index].set_title('') # remove plot title
        index = index + 1
        sns.distplot(df[x], kde=True, fit=norm, ax=ax[index])
        index = index + 1

printChart(all_data_sub_impute, num_col)
from scipy.stats import shapiro

# check up the target variable
SalePrice_statisitcs, SalePrice_p = shapiro(SalePrice)
print('data SalePrice does not look Gaussian (reject H0):', SalePrice_p < 0.05)

# check up the input features
def checkNormality(df, col):
    '''check normality for col and return non-normal distributed column names'''
    non_normal_col = []
    for x in col:
        statisitcs, p = shapiro(df[x])
        if p < 0.05:
            print('data {} does not look Gaussian (reject H0)'.format(x))
            non_normal_col.append(x)
        else:
            continue
    return non_normal_col

# for 'YearBuilt','YearRemodAdd','GarageYrBlt', remove them for boxcox transformation
# because their boxcox transformed values are too big and won't be able to serve as input for SVR, Decision Tree etc. 
boxcox_col = checkNormality(all_data_sub_impute, num_col)
boxcox_col.remove('YearBuilt')
boxcox_col.remove('YearRemodAdd')
boxcox_col.remove('GarageYrBlt')
# transform `SalePrice`
SalePrice_boxcox, SalePrice_maxlog = stats.boxcox(1 + SalePrice)

# implement normality test again
SalePrice_boxcox_statistics, SalePrice_boxcox_p = stats.shapiro(SalePrice_boxcox)
print('P value:', SalePrice_boxcox_p)
print('data SalePrice_boxcox does not look Gaussian (reject H0):', SalePrice_boxcox_p < 0.05)
print('SalePrice Skewness:', skew(SalePrice))
print('SalePrice_boxcox Skewness:', skew(SalePrice_boxcox))

fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.distplot(SalePrice, kde=True, fit=norm, ax=ax[0])
ax[0].set_title('Original SalePrice')
sns.distplot(SalePrice_boxcox, kde=True, fit=norm, ax=ax[1])
ax[1].set_title('Boxcox Transformed SalePrice')
# transform numerical input features
from pprint import pprint
def transBoxcox(df, boxcox_col):
    '''return boxcox transformed df and the lmbda of respective column for boxcox'''
    df = df.copy()
    boxcox_maxlog = {}
    for x in boxcox_col:
        array, maxlog = stats.boxcox(1 + df[x])
        df[x] = array
        boxcox_maxlog[x] = maxlog
    return df, boxcox_maxlog

all_data_sub_impute_boxcox , boxcox_maxlog = transBoxcox(all_data_sub_impute, boxcox_col)

# print subset data
pprint(dict((k,boxcox_maxlog[k]) for k in ('1stFlrSF','2ndFlrSF')))
all_data_sub_impute_boxcox.head(n=5)
# comparison of skewness of before and after on input features
skew_dataset = pd.DataFrame({'A_inputOriginalSkew':skew(all_data_sub_impute[boxcox_col]),
                             'B_inputBoxcoxSkew':skew(all_data_sub_impute_boxcox[boxcox_col])},
                            index=boxcox_col)
print(skew_dataset[:5])
# create dummy
all_data_sub_impute_boxcox_dummy = pd.get_dummies(all_data_sub_impute_boxcox)
all_data_sub_impute_boxcox_dummy.columns
# split out train and test dataset
train_impute_boxcox = all_data_sub_impute_boxcox_dummy[all_data_sub_impute_boxcox_dummy['DataSet_train'] == 1]
train_impute_boxcox.drop(columns=['DataSet_train','DataSet_test'], inplace=True)
test_impute_boxcox = all_data_sub_impute_boxcox_dummy[all_data_sub_impute_boxcox_dummy['DataSet_test'] == 1]
test_impute_boxcox.drop(columns=['DataSet_train','DataSet_test'], inplace=True)
print('train dataset shape:',train_impute_boxcox.shape)
print('test dataset shape:',test_impute_boxcox.shape)
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(train_impute_boxcox, SalePrice_boxcox, 
                                                    test_size = 0.2, random_state = 120)
# shape of new training and test dataset
print('new train dataset shape:',train_x.shape)
print('new test dataset shape:',test_x.shape)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
from sklearn.metrics import make_scorer, r2_score

def fitGridModel(model, params, train_x, train_y):
    '''define a function to implement GridSearch all in one'''
    reg_model = model
    best_params = {}
    
    scorer = make_scorer(r2_score, greater_is_better=True)
    grid = GridSearchCV(reg_model, param_grid=params, scoring=scorer, cv=5)
    grid.fit(train_x, train_y)
    score = grid.best_score_
    print("the best R^2 of all model parameters' combination on model: {:.4f}".format(score))
    best_params.update(grid.best_params_)
    print("the parameter setting of optimized model:", grid.best_estimator_)
    
    return score, best_params, grid
# Lasso
las_score, las_params, las_grid_model = fitGridModel(model=Lasso(), 
                                                     params = {"alpha":[0.1,0.001,0.0001,0.00005,0.00001],
                                                               "max_iter":[500,100,80,60],
                                                               "random_state":[120]},
                                                     train_x=train_x,
                                                     train_y=train_y)
# AdaBoost
adab_score, adab_params, adab_grid_model = fitGridModel(model=AdaBoostRegressor(),
                                                        params = {"n_estimators":[200,180,150,130,100],
                                                                  "learning_rate":[0.35,0.3,0.25,0.2,0.15],
                                                                  "random_state":[120]},
                                                        train_x = train_x,
                                                        train_y = train_y)
# SVR
svr_score, svr_params, svr_grid_model = fitGridModel(model=SVR(),
                                                     params = {'C':[7.5, 8, 8.5, 9, 9.5, 10],
                                                               'gamma':[0.000001, 0.00001, 0.0001, 0.001]},
                                                     train_x = train_x,
                                                     train_y = train_y)
# XGBoost
xgb_score, xgb_params, xgb_grid_model = fitGridModel(xgb.XGBRegressor(),
                                                     params = {'learning_rate': [0.65,0.7],
                                                               'max_depth': [1,2],
                                                               'colsample_bytree':[0.8,0.85],
                                                               'n_estimators':[300,350],
                                                               'random_state':[120]},
                                                     train_x = train_x,
                                                     train_y = train_y)
# collect the best params for each model
params = {Lasso.__name__:las_params,
          xgb.XGBRegressor.__name__: xgb_params,
          SVR.__name__: svr_params}
print(params)
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import KFold

class averageCVBaseModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y, params=None):
        self.models_ = []
        
        for m in self.models:
            if params is None:
                reg = m()
            else:
                params_value = params[m.__name__]
                reg = m(**params_value)
            reg.fit(X,y)
            self.models_.append(reg)
        
        return self
    
    def predict(self, X):
        predict_dataset = np.column_stack([reg.predict(X) for reg in self.models_])
        return np.mean(predict_dataset, axis = 1)
    
    def cvScore(self, X, y, n_folds, params=None):
        '''define to implement cross-validation for X and return r2 score'''
        kfold = KFold(n_splits=n_folds, shuffle=True)
        cv_scores = []
        
        for train_idx, valid_idx in kfold.split(X):
            train_x = X.iloc[train_idx,:]
            train_y = y[train_idx]
            valid_x = X.iloc[valid_idx,:]
            
            self.cv_models_ = []
            for m in self.models:
                if params is None:
                    reg = m()
                else:
                    params_value = params[m.__name__]
                    reg = m(**params_value)
                reg.fit(train_x,train_y)
                self.cv_models_.append(reg)
            
            predict_dataset = np.column_stack([reg.predict(valid_x) for reg in self.cv_models_])
            predict_dataset = np.mean(predict_dataset, axis = 1)
            r2 = r2_score(y[valid_idx], predict_dataset)
            cv_scores.append(r2)
        
        print('score in each fold:', cv_scores)
        return np.mean(cv_scores)
    
    def cvScoreThreeModelUnevenWeights(self, X, y, n_folds, params=None):
        '''define to implement cross-validation based on uneven weights
            for each model's prediction, 3 weights only, and return the mean
            cross-validation score under each weight set'''
        cv_scores_weighted = []
        for alpha in np.linspace(0.1, 0.9, 5, endpoint=False):
            for beta in np.linspace(0.1, (1-alpha), 5, endpoint=False):
                gamma = 1 - alpha - beta
        
                kfold = KFold(n_splits=n_folds, shuffle=True)
                cv_scores = []
        
                for train_idx, valid_idx in kfold.split(X):
                    train_x = X.iloc[train_idx,:]
                    train_y = y[train_idx]
                    valid_x = X.iloc[valid_idx,:]
            
                    self.cv_models_ = []
                    for m in self.models:
                        if params is None:
                            reg = m()
                        else:
                            params_value = params[m.__name__]
                            reg = m(**params_value)
                        reg.fit(train_x,train_y)
                        self.cv_models_.append(reg)
            
                    predict_dataset = np.column_stack([reg.predict(valid_x) for reg in self.cv_models_])
                
                    predict_dataset_weighted = alpha*predict_dataset[:,0] + \
                                               beta*predict_dataset[:,1] + \
                                               gamma*predict_dataset[:,2]
                    r2 = r2_score(y[valid_idx], predict_dataset_weighted)
                    cv_scores.append(r2)
                    
                cv_scores_weighted.append((alpha, beta, gamma, np.mean(cv_scores)))
        cv_scores_weighted = pd.DataFrame(cv_scores_weighted,
                                          columns=['alpha','beta','gamma','cv_score'])
        return cv_scores_weighted
        
# create an ensemble base model
avg_base_model = averageCVBaseModel(models = [Lasso, xgb.XGBRegressor, SVR])
# model with default parameter
avg_base_model.cvScore(X=train_x, y=train_y, n_folds=5,params=None)
# model with optimized parameters
avg_base_model.cvScore(X=train_x, y=train_y, n_folds=5,params=params)
# find out the best weight set for ensemble base model
best_weight_set = avg_base_model.cvScoreThreeModelUnevenWeights(X=train_x, 
                                                                y=train_y,
                                                                n_folds=5,
                                                                params=params)
print('top five weight set for cv score: \n', 
      best_weight_set.iloc[np.argsort(best_weight_set['cv_score'])][::-1][:5])
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

RANDOM_SEED = 120

# The StackingCVRegressor uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior

# use the optimized parameters from above
lasso = Lasso(random_state=RANDOM_SEED, alpha=0.0001, max_iter=80)
xgb_model = xgb.XGBRegressor(random_state = RANDOM_SEED, colsample_bytree=0.85,
                             learning_rate=0.7, max_depth=1, 
                             n_estimators=350)
svr = SVR(C=9.5, gamma=0.00001)
linear = LinearRegression() # meta-model

np.random.seed(RANDOM_SEED)
stack = StackingCVRegressor(regressors = (lasso, xgb_model, svr),
                            meta_regressor = linear)


stack_scores = cross_val_score(stack, train_x.as_matrix(), train_y, cv=5)

print("R^2 Score: %0.2f (+/- %0.2f)" %(stack_scores.mean(), stack_scores.std()))
'''
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

RANDOM_SEED = 120

# The StackingCVRegressor uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior

# use the optimized parameters from above
lasso = Lasso(random_state=RANDOM_SEED)
xgb_model = xgb.XGBRegressor(random_state = RANDOM_SEED)
svr = SVR()
linear = LinearRegression() # meta-model

np.random.seed(RANDOM_SEED)
grid_stack = StackingCVRegressor(regressors = (lasso, xgb_model, svr),
                                 meta_regressor = linear)

# the code for grid sreach
# for it taks up so much time, I will not run this model, but it serves as a document
params_mlx = {"lasso__alpha":[0.1,0.001,0.0001,0.00005,0.00001],
              "lasso__max_iter":[500,100,80,60],
              "xgbregressor__learning_rate": [0.65,0.7,0.75,0.8,0.85],
              "xgbregressor__max_depth": [1,2,3],
              "xgbregressor__colsample_bytree":[0.8,0.85, 0.9],
              "xgbregressor__n_estimators":[300,350,400],
              "svr__C":[7.5, 8, 8.5, 9, 9.5, 10],
              "svr__gamma":[0.000001, 0.00001, 0.0001, 0.001]}

grid_mlx = GridSearchCV(estimator=grid_stack,
                        param_grid = params_mlx,
                        cv = 5)


# both X,y should be numpy.array
grid_mlx.fit(train_x.as_matrix(), train_y)

print("Best: %f using %s" % (grid_mlx.best_score_, grid_mlx.best_params_))

'''
# use named_regressors to get `name` of the regressor
print(stack.named_regressors)
print(stack.named_meta_regressor)
# get parameter key for the model
AdaBoostRegressor().get_params().keys()
# XGBoost model on test dataset
prediction = xgb_grid_model.predict(test_x)
print('single model XGBoost R2:',r2_score(test_y, prediction))
from scipy.special import inv_boxcox
print('sale price boxcox lmbda:',SalePrice_maxlog)
outcome = pd.DataFrame(np.column_stack((inv_boxcox(test_y, SalePrice_maxlog), 
                                        inv_boxcox(prediction, SalePrice_maxlog))),
                       columns = ['original_y', 'predicted_y'])
print('R2 on original scale:',r2_score(outcome.original_y, outcome.predicted_y))
avg_base_model.fit(train_x, train_y)
prediction = avg_base_model.predict(test_x)
print('Average Base Model R2:',r2_score(test_y, prediction))
outcome = pd.DataFrame(np.column_stack((inv_boxcox(test_y, SalePrice_maxlog), 
                                        inv_boxcox(prediction, SalePrice_maxlog))),
                       columns = ['original_y', 'predicted_y'])
print('R2 on original scale:',r2_score(outcome.original_y, outcome.predicted_y))
avg_base_model.fit(train_x, train_y, params=params)
prediction = avg_base_model.predict(test_x)
print('Average Base Model with Optimized Parameters R2:',r2_score(test_y, prediction))
outcome = pd.DataFrame(np.column_stack((inv_boxcox(test_y, SalePrice_maxlog), 
                                        inv_boxcox(prediction, SalePrice_maxlog))),
                       columns = ['original_y', 'predicted_y'])
print('R2 on original scale:',r2_score(outcome.original_y, outcome.predicted_y))
stack.fit(train_x.as_matrix(), train_y)
prediction = stack.predict(test_x.as_matrix())
print('Stacking Meta Model with Optimized Parameters R2:',r2_score(test_y, prediction))
outcome = pd.DataFrame(np.column_stack((inv_boxcox(test_y, SalePrice_maxlog), 
                                        inv_boxcox(prediction, SalePrice_maxlog))),
                       columns = ['original_y', 'predicted_y'])
print('R2 on original scale:',r2_score(outcome.original_y, outcome.predicted_y))
# have a look at y in original sacle
outcome.head(n=10)
# calculate the mean squared error
from sklearn.metrics import mean_squared_error
print('square root of mean squared error',math.sqrt(mean_squared_error(outcome.original_y, outcome.predicted_y)))
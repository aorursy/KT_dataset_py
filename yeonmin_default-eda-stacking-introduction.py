# Loading packages

import pandas as pd #Analysis 

import matplotlib.pyplot as plt #Visulization

import seaborn as sns #Visulization

import numpy as np #Analysis 

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats #Analysis 

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc



import lightgbm as lgb

import xgboost as xgb

import catboost as cb



import time

from datetime import datetime, timedelta,date



from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import ElasticNet, Lasso, Ridge

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.preprocessing import RobustScaler
def check_train_test_diff(train, test, col):

    ratio_df = pd.concat([train[col].value_counts()/train.shape[0], test[col].value_counts()/test.shape[0]], axis=1)

    ratio_df.columns = ['train','test']

    ratio_df['diff'] = ratio_df['train'] - ratio_df['test']

    return ratio_df
def category_feature_distribution(train, col, target='price'):

    fig, ax = plt.subplots(1, 2, figsize=(16,4))

    

    for c in sorted(train[col].unique()):

        sns.distplot(np.log1p(train.loc[train[col]==c, target]), ax=ax[0])

    ax[0].legend(sorted(train[col].unique()))

    ax[0].set_title(f'{col} {target} distribution')



    sns.boxplot(x=col, y=target, data=df_train, ax=ax[1])

    ax[1].set_title(f'{col} vs {target}')

    

    plt.show()
def continous_feature_distribution(train, test, col, target='price'):

    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    

    sns.distplot(train[col], ax=ax[0])

    sns.distplot(test[col], ax=ax[0])

    ax[0].set_title(f'{col} - train/test distribution')

    

    sns.scatterplot(x=col, y=target, data=train, ax=ax[1])

    ax[1].set_title(f'{col} - {target} scatterplot')

    

    plt.show()
def scatter_quantile_graph(frame, col1, col2):

    col1_quantile = np.arange(0,1.1,0.1)

    col2_quantile = np.arange(0,1.1,0.1)



    for quantile_value in frame[col1].quantile(col1_quantile):

        plt.axvline(quantile_value, color='red', alpha=0.3)

    for quantile_value in frame[col2].quantile(col2_quantile):

        plt.axhline(quantile_value, color='blue', alpha=0.3)

        

    sns.scatterplot(col1, col2, hue='price',data=frame)

    

    plt.title('{} - {}'.format(col1,col2))

    plt.show()
def get_prefix(group_col, target_col, prefix=None):

    if isinstance(group_col, list) is True:

        g = '_'.join(group_col)

    else:

        g = group_col

    if isinstance(target_col, list) is True:

        t = '_'.join(target_col)

    else:

        t = target_col

    if prefix is not None:

        return prefix + '_' + g + '_' + t

    return g + '_' + t

    

def groupby_helper(df, group_col, target_col, agg_method, prefix_param=None):

    try:

        prefix = get_prefix(group_col, target_col, prefix_param)

        print(group_col, target_col, agg_method)

        group_df = df.groupby(group_col)[target_col].agg(agg_method)

        group_df.columns = ['{}_{}'.format(prefix, m) for m in agg_method]

    except BaseException as e:

        print(e)

    return group_df.reset_index()
from functools import wraps

def time_decorator(func): 

    @wraps(func)

    def wrapper(*args, **kwargs):

        print("\nStartTime: ", datetime.now() + timedelta(hours=9))

        start_time = time.time()

        

        df = func(*args, **kwargs)

        

        print("EndTime: ", datetime.now() + timedelta(hours=9))  

        print("TotalTime: ", time.time() - start_time)

        return df

        

    return wrapper



class SklearnWrapper(object):

    def __init__(self, clf, params=None, **kwargs):

        #if isinstance(SVR) is False:

        #    params['random_state'] = kwargs.get('seed', 0)

        self.clf = clf(**params)

        self.is_classification_problem = True

    @time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        if len(np.unique(y_train)) > 30:

            self.is_classification_problem = False

            

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        if self.is_classification_problem is True:

            return self.clf.predict_proba(x)[:,1]

        else:

            return self.clf.predict(x)   

    

class XgbWrapper(object):

    def __init__(self, params=None, **kwargs):

        self.param = params

        self.param['seed'] = kwargs.get('seed', 0)

        self.num_rounds = kwargs.get('num_rounds', 1000)

        self.early_stopping = kwargs.get('ealry_stopping', 100)



        self.eval_function = kwargs.get('eval_function', None)

        self.verbose_eval = kwargs.get('verbose_eval', 100)

        self.best_round = 0

    

    @time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        need_cross_validation = True

       

        if isinstance(y_train, pd.DataFrame) is True:

            y_train = y_train[y_train.columns[0]]

            if y_cross is not None:

                y_cross = y_cross[y_cross.columns[0]]



        if x_cross is None:

            dtrain = xgb.DMatrix(x_train, label=y_train, silent= True)

            train_round = self.best_round

            if self.best_round == 0:

                train_round = self.num_rounds

            

            print(train_round)

            self.clf = xgb.train(self.param, dtrain, train_round)

            del dtrain

        else:

            dtrain = xgb.DMatrix(x_train, label=y_train, silent=True)

            dvalid = xgb.DMatrix(x_cross, label=y_cross, silent=True)

            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]



            self.clf = xgb.train(self.param, dtrain, self.num_rounds, watchlist, feval=self.eval_function,

                                 early_stopping_rounds=self.early_stopping,

                                 verbose_eval=self.verbose_eval)

            self.best_round = max(self.best_round, self.clf.best_iteration)



    def predict(self, x):

        return self.clf.predict(xgb.DMatrix(x), ntree_limit=self.best_round)



    def get_params(self):

        return self.param    

    

class LgbmWrapper(object):

    def __init__(self, params=None, **kwargs):

        self.param = params

        self.param['seed'] = kwargs.get('seed', 0)

        self.num_rounds = kwargs.get('num_rounds', 1000)

        self.early_stopping = kwargs.get('ealry_stopping', 100)



        self.eval_function = kwargs.get('eval_function', None)

        self.verbose_eval = kwargs.get('verbose_eval', 100)

        self.best_round = 0

        

    @time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        """

        x_cross or y_cross is None

        -> model train limted num_rounds

        

        x_cross and y_cross is Not None

        -> model train using validation set

        """

        if isinstance(y_train, pd.DataFrame) is True:

            y_train = y_train[y_train.columns[0]]

            if y_cross is not None:

                y_cross = y_cross[y_cross.columns[0]]



        if x_cross is None:

            dtrain = lgb.Dataset(x_train, label=y_train, silent= True)

            train_round = self.best_round

            if self.best_round == 0:

                train_round = self.num_rounds

                

            self.clf = lgb.train(self.param, train_set=dtrain, num_boost_round=train_round)

            del dtrain   

        else:

            dtrain = lgb.Dataset(x_train, label=y_train, silent=True)

            dvalid = lgb.Dataset(x_cross, label=y_cross, silent=True)

            self.clf = lgb.train(self.param, train_set=dtrain, num_boost_round=self.num_rounds, valid_sets=[dtrain, dvalid],

                                  feval=self.eval_function, early_stopping_rounds=self.early_stopping,

                                  verbose_eval=self.verbose_eval)

            self.best_round = max(self.best_round, self.clf.best_iteration)

            del dtrain, dvalid

            

        gc.collect()

    

    def predict(self, x):

        return self.clf.predict(x, num_iteration=self.clf.best_iteration)

    

    def plot_importance(self):

        lgb.plot_importance(self.clf, max_num_features=50, height=0.7, figsize=(10,30))

        plt.show()

        

    def get_params(self):

        return self.param

    

    

@time_decorator

def get_oof(clf, x_train, y_train, x_test, eval_func, **kwargs):

    nfolds = kwargs.get('NFOLDS', 5)

    kfold_shuffle = kwargs.get('kfold_shuffle', True)

    kfold_random_state = kwargs.get('kfold_random_state', 0)

    stratified_kfold_ytrain = kwargs.get('stratifed_kfold_y_value', None)

    ntrain = x_train.shape[0]

    ntest = x_test.shape[0]

    

    kf_split = None

    if stratified_kfold_ytrain is None:

        kf = KFold(n_splits=nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)

        kf_split = kf.split(x_train)

    else:

        kf = StratifiedKFold(n_splits=nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)

        kf_split = kf.split(x_train, stratified_kfold_ytrain)

        

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))



    cv_sum = 0

    

    # before running model, print model param

    # lightgbm model and xgboost model use get_params()

    try:

        if clf.clf is not None:

            print(clf.clf)

    except:

        print(clf)

        print(clf.get_params())



    for i, (train_index, cross_index) in enumerate(kf_split):

        x_tr, x_cr = None, None

        y_tr, y_cr = None, None

        if isinstance(x_train, pd.DataFrame):

            x_tr, x_cr = x_train.iloc[train_index], x_train.iloc[cross_index]

            y_tr, y_cr = y_train.iloc[train_index], y_train.iloc[cross_index]

        else:

            x_tr, x_cr = x_train[train_index], x_train[cross_index]

            y_tr, y_cr = y_train[train_index], y_train[cross_index]



        clf.train(x_tr, y_tr, x_cr, y_cr)

        

        oof_train[cross_index] = clf.predict(x_cr)



        cv_score = eval_func(y_cr, oof_train[cross_index])

        

        print('Fold %d / ' % (i+1), 'CV-Score: %.6f' % cv_score)

        cv_sum = cv_sum + cv_score

        

        del x_tr, x_cr, y_tr, y_cr

        

    gc.collect()

    

    score = cv_sum / nfolds

    print("Average CV-Score: ", score)



    # Using All Dataset, retrain

    clf.train(x_train, y_train)

    oof_test = clf.predict(x_test)



    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1), score
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
print("train.csv. Shape: ",df_train.shape)

print("test.csv. Shape: ",df_test.shape)
df_train.head()
default_check = pd.concat([df_train.isnull().sum(), df_train.dtypes, df_train.nunique(), df_train.describe().T], axis=1)

default_check.rename(columns={0:'NULL', 1:'TYPE', 2:'UNIQUE'}, inplace=True)

default_check
fig, ax = plt.subplots(1, 3, figsize=(20,4))

sns.distplot(df_train['price'], ax=ax[0])

sns.distplot(np.log1p(df_train['price']), ax=ax[1])

df_train['price'].plot(ax=ax[2])

plt.show()
from IPython.display import display

for col in ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade']:

    print(col)

    display(check_train_test_diff(df_train, df_test, col))
for col in ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade']:

    category_feature_distribution(df_train, col)
df_train['date'] = pd.to_datetime(df_train['date'])

print(df_train['date'].min(), df_train['date'].max())



df_test['date'] = pd.to_datetime(df_test['date'])

print(df_test['date'].min(), df_test['date'].max())
print("Train")

display(df_train.sort_values('date').head())

print("Test")

display(df_test.sort_values('date').head())
plt.plot(df_train.sort_values('date')['price'].cumsum().values)
fig, ax = plt.subplots(1, 4, figsize=(20,5))

df_train.groupby('date')['price'].count().plot(ax=ax[0])

ax[0].set_title('Each date price count')

df_train.groupby('date')['price'].sum().plot(ax=ax[1])

ax[1].set_title('Each date price sum')

df_train.groupby('date')['price'].mean().plot(ax=ax[2])

ax[2].set_title('Each date price mean')

df_train.groupby('date')['price'].std().plot(ax=ax[3])

ax[3].set_title('Each date price std')

plt.show()
df_train.loc[df_train['date']==np.argmax(df_train.groupby('date')['price'].mean())]
fig, ax = plt.subplots(1, 2, figsize=(10,5))

df_train.groupby('date')['id'].count().plot(ax=ax[0])

ax[0].set_title('Train each date sales count')

df_test.groupby('date')['id'].count().plot(ax=ax[1])

ax[1].set_title('Test each date sales count')

plt.show()
df_train['yearmonth'] = df_train['date'].dt.year*100 + df_train['date'].dt.month

category_feature_distribution(df_train,'yearmonth')

area_feature = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

for col in area_feature:

    continous_feature_distribution(df_train, df_test, col)
sns.scatterplot('long','lat',hue='price',data=df_train)
scatter_quantile_graph(df_train, 'long', 'lat')
qcut_count = 10

df_train['qcut_long'] = pd.qcut(df_train['long'], qcut_count, labels=range(qcut_count))

df_train['qcut_lat'] = pd.qcut(df_train['lat'], qcut_count, labels=range(qcut_count))

temp = df_train.groupby(['qcut_long','qcut_lat'])['price'].mean().reset_index()

sns.scatterplot('qcut_long','qcut_lat', hue='price', data=temp);

del df_train['qcut_long'], df_train['qcut_lat']
df_train = df_train.loc[df_train['bedrooms']<10]
skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    df_train[c] = np.log1p(df_train[c].values)

    df_test[c] = np.log1p(df_test[c].values)
for df in [df_train,df_test]:

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])
def feature_processing(df):

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    df['grade_condition'] = df['grade'] * df['condition']

    df['sqft_total'] = df['sqft_living'] + df['sqft_lot']

    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']

   

    df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15'] 

    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)

    

    df['roombybathroom'] = df['bedrooms'] / df['bathrooms']

    df['sqft_total_by_lot'] = (df['sqft_living'] + df['sqft_above'] + df['sqft_basement'])/df['sqft_lot']

    

    qcut_count = 10

    df['qcut_long'] = pd.qcut(df['long'], qcut_count, labels=range(qcut_count))

    df['qcut_lat'] = pd.qcut(df['lat'], qcut_count, labels=range(qcut_count))

    df['qcut_long'] = df['qcut_long'].astype(int)

    df['qcut_lat'] = df['qcut_lat'].astype(int)



    df['date'] = pd.to_datetime(df['date'])

    df['yearmonth'] = df['date'].dt.year*100 + df['date'].dt.month

    df['date'] = df['date'].astype('int')

    return df
all_df = pd.concat([df_train, df_test])

all_df = feature_processing(all_df)
df_test = all_df.loc[all_df['price'].isnull()]

df_train = all_df.loc[all_df['price'].notnull()]
group_df = groupby_helper(df_train, 'grade', 'price', ['mean'])

df_train = df_train.merge(group_df, on='grade', how='left')

df_test = df_test.merge(group_df, on='grade', how='left')



group_df = groupby_helper(df_train, 'bedrooms', 'price', ['mean'])

df_train = df_train.merge(group_df, on='bedrooms', how='left')

df_test = df_test.merge(group_df, on='bedrooms', how='left')



group_df = groupby_helper(df_train, 'bathrooms', 'price', ['mean'])

df_train = df_train.merge(group_df, on='bathrooms', how='left')

df_test = df_test.merge(group_df, on='bathrooms', how='left')
train_columns = [col for col in df_train.columns if col not in ['id','price']]



x_train = df_train.copy()

y_train = np.log1p(df_train['price'])

del x_train['price']

x_train.loc[np.isinf(x_train['roombybathroom']),'roombybathroom'] = -1





x_test = df_test.copy()

x_test.loc[np.isinf(x_test['roombybathroom']),'roombybathroom'] = -1
def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))
lgb_param = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.015,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 4950}



xgb_params = {

    'eval_metric': 'rmse',

    'seed': 4950,

    'eta': 0.0123,

    'gamma':0,

    'max_depth':3,

    'reg_alpha':0.00006,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'silent': 1,

}



ridge_params = {'alpha':0.0001, 'normalize':True, 'max_iter':1e7, 'random_state':0}

lasso_params = {'alpha':0.0005, 'normalize':True, 'max_iter':1e7, 'random_state':0}

elastic_params = {'alpha':0.001, 'normalize':True, 'max_iter':1e3, 'random_state':0, 'l1_ratio':0.8}

svr_param = {'C':20, 'epsilon':0.008, 'gamma':0.0003}

gbr_param = {'n_estimators':3000, 'learning_rate':0.05, 'max_depth':4, 'max_features':'sqrt', 'min_samples_leaf':15, 'min_samples_split':10, 'loss':'huber', 'random_state':0 }
xgb_model = XgbWrapper(params=xgb_params, num_rounds = 10000, ealry_stopping=100,

                                   verbose_eval=100)



lgb_model = LgbmWrapper(params=lgb_param, num_rounds = 10000, ealry_stopping=100,

                                   verbose_eval=100)



ridge_model = SklearnWrapper(Ridge, params=ridge_params)

lasso_model = SklearnWrapper(Lasso, params=lasso_params)

elastic_model = SklearnWrapper(ElasticNet, params=lasso_params)

svr_model = SklearnWrapper(SVR, params=svr_param)

gbr_model = SklearnWrapper(GradientBoostingRegressor, params=gbr_param)
x_train_rb = x_train.copy()

x_test_rb = x_test.copy()

rb = RobustScaler()

x_train_rb[train_columns] = rb.fit_transform(x_train_rb[train_columns].fillna(-1))

x_test_rb[train_columns] = rb.transform(x_test_rb[train_columns].fillna(-1))
ridge_train, ridge_test, ridge_cv_score = get_oof(ridge_model, x_train_rb[train_columns], y_train, x_test_rb[train_columns], 

                            rmse, NFOLDS=5, kfold_random_state=4950)



lasso_train, lasso_test, lasso_cv_score = get_oof(lasso_model, x_train_rb[train_columns], y_train, x_test_rb[train_columns], 

                            rmse, NFOLDS=5, kfold_random_state=4950)



elastic_train, elastic_test, lasso_cv_score = get_oof(elastic_model, x_train_rb[train_columns], y_train, x_test_rb[train_columns], 

                            rmse, NFOLDS=5, kfold_random_state=4950)



svr_train, svr_test, lasso_cv_score = get_oof(svr_model, x_train_rb[train_columns], y_train, x_test_rb[train_columns], 

                            rmse, NFOLDS=5, kfold_random_state=4950)



gbr_train, gbr_test, lasso_cv_score = get_oof(gbr_model, x_train[train_columns].fillna(-1), y_train, x_test[train_columns].fillna(-1), 

                            rmse, NFOLDS=5, kfold_random_state=4950)



xgb_train, xgb_test, xgb_cv_score = get_oof(xgb_model, x_train[train_columns], y_train, x_test[train_columns], 

                            rmse, NFOLDS=5, kfold_random_state=4950)



lgb_train, lgb_test, lgb_cv_score = get_oof(lgb_model, x_train[train_columns], y_train, x_test[train_columns], 

                            rmse, NFOLDS=5, kfold_random_state=4950)
x_train_second_layer = np.concatenate((lgb_train, xgb_train, lasso_train, 

                                       ridge_train, elastic_train, svr_train, 

                                       gbr_train), axis=1)



x_test_second_layer = np.concatenate((lgb_test, xgb_test, lasso_test, 

                                      ridge_test, elastic_test, svr_test, 

                                      gbr_test), axis=1)



x_train = pd.concat([df_train['id'], pd.DataFrame(x_train_second_layer)], axis=1)

x_test = pd.concat([df_test['id'], pd.DataFrame(x_test_second_layer)], axis=1)



x_train.to_csv('train_oof.csv', index=False)

x_test.to_csv('test_oof.csv', index=False)

del x_train['id']

del x_test['id']
lgb_meta_param = {'num_leaves': 15,

         'objective':'regression',

         'max_depth': 5,

         'learning_rate': 0.015,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 4950}

         

#prepare fit model with cross-validation

folds = KFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(x_train))

predictions = np.zeros(len(x_test))

feature_importance_df = pd.DataFrame()



#run model

for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train)):

    trn_data = lgb.Dataset(x_train.iloc[trn_idx], label=y_train.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(x_train.iloc[val_idx], label=y_train.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 10000

    clf = lgb.train(lgb_meta_param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 100)

    oof[val_idx] = clf.predict(x_train.iloc[val_idx], num_iteration=clf.best_iteration)

   

    #predictions

    predictions += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits

    

cv = np.sqrt(mean_squared_error(oof, y_train))

print(cv)
submission = pd.DataFrame({'id': df_test['id'], 'price': np.expm1(predictions)})

submission.to_csv('submission.csv', index=False)
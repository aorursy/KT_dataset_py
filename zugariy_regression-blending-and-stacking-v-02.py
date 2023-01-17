import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew



from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LassoCV

from sklearn.svm import SVC

from sklearn.neighbors import LocalOutlierFactor



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline



pd.options.display.max_rows = 10

pd.options.display.float_format = '{:.1f}'.format



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train_ID = train.Id

test_ID = test.Id



n_target = train.SalePrice



_=train.pop('Id')

_=test.pop('Id')
def show_dist(x):

    sns.distplot(x, fit=norm)

    (mu, sigma) = norm.fit(x)



    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)], loc='best')

    plt.ylabel('Frequency')

    plt.title('SalePrice distribution')



    fig = plt.figure()

    res = stats.probplot(x, plot=plt)

    plt.show()

    print("Skewness : %.2f" % x.skew())

    print("Kurtosis : %.2f" % x.kurt())

    return
show_dist(n_target)
target = np.log(n_target)

show_dist(target)
def na_count(df):

    total = df.isna().sum().sort_values(ascending=False)

    percent = 100*(total/df.shape[0])

    return pd.concat([total, percent], axis=1, keys=['Total NA', '%'])
na_count(train).head(10)
def reg_on_na(ser,target, p=0.25):

    n_missing = ser.isna().sum()

    percent = n_missing/ser.shape[0]

    if n_missing == 0:

        print("no missing values in :"+str(ser.name))

    else:    

        if percent < p :

            # missing value index

            m_id = ser[ser.isna()].index

            # labels : non-NA values of our missing series

            Y = np.array(ser.drop(m_id)).reshape(-1,1)

            # target values (salerice) that we will train to predict missing feature

            # Single feature data must be reshaped before training

            X = np.array(target.drop(m_id)).reshape(-1,1)

            # Missing saleprices upon which we will make prediction

            Xm = np.array(target[m_id]).reshape(-1,1)

            reg = LassoCV(cv=5, random_state=0).fit(X,Y)

            ser[m_id] = reg.predict(Xm)

        else :

            print("You should drop :"+str(ser.name))   

    return ser



def class_on_na(ser,target, p=0.25):

    n_missing = ser.isna().sum()

    percent = n_missing/ser.shape[0]

    if n_missing == 0:

        print("no missing values in :"+str(ser.name))

    else:    

        if percent < p :

            # missing value index

            m_id = ser[ser.isna()].index

            # labels : non-NA values of our missing series

            Y = np.array(ser.drop(m_id)).reshape(-1,1)

            # target values (salerice) that we will train to predict missing feature

            # Single feature data must be reshaped before training

            X = np.array(target.drop(m_id)).reshape(-1,1)

            # Missing saleprices upon which we will make prediction

            Xm = np.array(target[m_id]).reshape(-1,1)

            clas = SVC(gamma=2, C=1).fit(X,Y)

            ser[m_id] = clas.predict(Xm)

        else :

            print("You should drop :"+str(ser.name))   

    return ser
def fill_missing(df, target=target):

    num_cols = df.select_dtypes([np.number]).columns

    cat_cols = df.select_dtypes([np.object]).columns

    dfnum = df[num_cols].copy()

    dfcat = df[cat_cols].copy()

    num_cols_miss = dfnum.isna().sum()[dfnum.isna().sum()>0].index

    cat_cols_miss = dfcat.isna().sum()[dfcat.isna().sum()>0].index

    df[num_cols_miss] = df[num_cols_miss].apply(lambda x: reg_on_na(x, target))

    df[cat_cols_miss] = df[cat_cols_miss].apply(lambda x: class_on_na(x, target))

    return df
train = fill_missing(train,target)



test.Utilities.fillna('AllPub', inplace=True)

test = fill_missing(test, test.GrLivArea)

def outliers(x, y=n_target, top=5, plot=True):

    lof = LocalOutlierFactor(n_neighbors=40, contamination=0.1)

    x_ =np.array(x).reshape(-1,1)

    preds = lof.fit_predict(x_)

    lof_scr = lof.negative_outlier_factor_

    out_idx = pd.Series(lof_scr).sort_values()[:top].index

    if plot:

        f, ax = plt.subplots(figsize=(9, 6))

        plt.scatter(x=x, y=y, c=np.exp(lof_scr), cmap='RdBu')

    return out_idx

    
outs = outliers(train['GrLivArea'], top=5)

train = train.drop(outs)

target = target.drop(outs)

n_target = n_target.drop(outs)

ntrain = train.shape[0]

ntest = test.shape[0]

alldata = pd.concat([train, test]).reset_index(drop=True)



alldata.drop(['SalePrice','Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

print(alldata.shape)

show_dist(target)
# All the feature engeneering i've got

alldata['TotalSF'] = alldata['TotalBsmtSF'] + alldata['1stFlrSF'] + alldata['2ndFlrSF']

alldata['Area_Qual'] = alldata['TotalSF']*alldata['OverallQual']

numeric_feats = alldata.select_dtypes([np.number]).columns



# Check the skew of all numerical features

skewed_feats = alldata[numeric_feats].apply(skew).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.10

_skewed = alldata[numeric_feats].apply(lambda x: boxcox1p(x,lam)).apply(skew).sort_values(ascending=False)

skewness['boxed'] = pd.Series(_skewed)

alldata[numeric_feats] = alldata[numeric_feats].apply(lambda x: boxcox1p(x,lam))

skewness.head(10)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, Ridge, SGDRegressor

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR, LinearSVR 

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV

import xgboost as xgb

import lightgbm as lgb

from mlxtend.regressor import StackingCVRegressor



import time
alldata = pd.get_dummies(alldata)

alldata = RobustScaler().fit_transform(alldata) 

alldata = PCA(n_components=0.999).fit_transform(alldata) 

train = alldata[:ntrain]

test = alldata[ntrain:]

y_train = n_target.values

print(train.shape)

print(test.shape)
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)

    rmse= np.sqrt(-cross_val_score(model, train, target.values, scoring="neg_mean_squared_error", cv = kf))

    return rmse
class grid():

    def __init__(self,model):

        self.model = model

    

    def grid_get(self,X,y,param_grid):

        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)

        grid_search = GridSearchCV(self.model,param_grid,cv=kf, scoring="neg_mean_squared_error")

        grid_search.fit(X,y)

        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))

        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])

        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
pd.options.display.float_format = '{:.4f}'.format

grid( Lasso(random_state=1)).grid_get(train,target.values,{'alpha': [0.0004,0.0005,0.0007,0.0009],'max_iter':[10000, 15000]})
# grid(lgb.LGBMRegressor(objective='regression',

#                               max_bin = 55, bagging_fraction = 0.8,

#                               bagging_freq = 5, feature_fraction = 0.2319,

#                               feature_fraction_seed=9, bagging_seed=9,

#                               min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)).grid_get(train,target.values,{'num_leaves': [5, 10 ,15],

#                                                                                                                 'learning_rate': [0.05, 0.01, 0.001],

#                                                                                                                'n_estimators': [500, 720, 900]})
grid(ElasticNet(random_state=3)).grid_get(train,target.values,{'alpha': [0.0004,0.0005,0.0007,0.001], 'l1_ratio': [0.3, 0.6, 0.9]})
grid(KernelRidge()).grid_get(train,target.values,{'alpha': [0.1,0.3,0.6,0.9], 'kernel':['linear','polynomial'], 'degree':[2,3,4], 'coef0':[0.01,2.5,5]})
grid(Ridge(random_state=5)).grid_get(train,target.values,{'alpha': [5,10,20,30]})
grid(SVR()).grid_get(train,target.values,{'gamma': [0.00001,0.00005,0.0009], 'epsilon': [0.001,0.005,0.009]})
models = [ Lasso(alpha =0.0005, random_state=1), 

         ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3), 

         KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5), 

         Ridge(alpha=20), 

         SVR(gamma= 0.0009,kernel='rbf',C=13,epsilon=0.005),

         BayesianRidge(),

         LinearSVR(),

         SGDRegressor(max_iter=1000,tol=1e-3)

         ]

names = ['Lasso', 'ElasticNet', 'KernelRidge', 'Ridge', 'SVR', 'BayesianRidge', 'LinearSVR', 'SGDRegressor']
for name, model in zip(names, models):

    start = time.time()

    score = rmsle_cv(model)

    end = time.time()

    print("{}: {:.6f}, {:.4f} in {:.3f} s".format(name,score.mean(),score.std(),end-start))
class AverageWeight(BaseEstimator, RegressorMixin):

    def __init__(self,mod,weight):

        self.mod = mod

        self.weight = weight

        

    def fit(self,X,y):

        self.models_ = [clone(x) for x in self.mod]

        for model in self.models_:

            model.fit(X,y)

        return self

    

    def predict(self,X):

        w = list()

        pred = np.array([model.predict(X) for model in self.models_])

        # for every data point, single model prediction times weight, then add them together

        for data in range(pred.shape[1]):

            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]

            w.append(np.sum(single))

        return w
sel_models = [Lasso(alpha =0.0005, random_state=1, max_iter=10000), 

             ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3), 

             KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5), 

             Ridge(alpha=20), 

             SVR(gamma= 0.0009,kernel='rbf',C=13,epsilon=0.005),

             BayesianRidge(),

             LinearSVR(),

              ]



 
np.random.seed(42)

stack = StackingCVRegressor(regressors=sel_models,

                            meta_regressor=SVR(gamma= 0.0009,kernel='rbf',C=13,epsilon=0.005),

                            use_features_in_secondary=True)

start = time.time()

score = rmsle_cv(stack)

end = time.time()

print("Stacked : {:.6f}, (+/-) {:.4f} in {:.3f} s".format(score.mean(),score.std(),end-start))
sel_models = [

             KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5), 

             Ridge(alpha=20), 

             SVR(gamma= 0.0009,kernel='rbf',C=13,epsilon=0.005),

             BayesianRidge(),

             stack

              ]



weights = [0.2,0.15,0.2,0.15,0.3]



start = time.time()

blended = AverageWeight(mod = sel_models ,weight=weights)

score = rmsle_cv(blended)

end = time.time()

print("Weighted avg: {:.6f}, {:.4f} in {:.3f} s".format(score.mean(),score.std(),end-start))
blended.fit(train, target.values)
pred = np.exp(blended.predict(test))

results = pd.DataFrame({'Id':test_ID, 'SalePrice':pred})

q1 = results['SalePrice'].quantile(0.0042)

q2 = results['SalePrice'].quantile(0.99)



results['SalePrice'] = results['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

results['SalePrice'] = results['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

results.to_csv("submission.csv",index=False)
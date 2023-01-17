!pip install bayesian-optimization


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os



color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

np.random.RandomState(seed=1)



train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

train_Id=train.Id

test_Id=test.Id

train=train.drop(['Id'],axis=1)

test=test.drop(['Id'],axis=1)
train.head()
train.describe()
train.info()
f, ax = plt.subplots(figsize=(15, 12))

corr=train.corr()

sns.heatmap(corr)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


train["SalePrice"] = np.log1p(train["SalePrice"])

sns.distplot(train.SalePrice)
sns.pairplot(train[train.dtypes[train.dtypes!="object"].index])
all_data=pd.concat([train,test])

missing=(all_data.isnull().sum().sort_values(ascending=False)/len(all_data))[:35]

missing_data = pd.DataFrame({'Missing Ratio' :missing})

missing_data.head(35)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=missing.index,y=missing)

plt.show()
all_data[missing.index].head(40)
for cols in ('MiscFeature','PoolQC','Alley','Fence','BsmtFinType1','KitchenQual','BsmtCond','BsmtQual','BsmtExposure','BsmtFinType2','GarageQual','GarageFinish','GarageCond','GarageType','FireplaceQu','MasVnrType','Exterior2nd'):

    all_data[cols]=all_data[cols].fillna("None")
for cols in ('GarageYrBlt','MasVnrArea','BsmtHalfBath','BsmtFullBath','GarageCars','GarageArea','TotalBsmtSF','BsmtUnfSF','BsmtFinSF2','BsmtFinSF1'):

    all_data[cols]=all_data[cols].fillna(0)
for cols in ('MSZoning','Electrical','Exterior2nd','Exterior1st','SaleType'):

    all_data[cols]=all_data[cols].fillna(all_data[cols].mode()[0])

    


all_data=all_data.drop(['Utilities'],axis=1)

all_data['Functional']=all_data['Functional'].fillna('Typ')

all_data['LotFrontage']=all_data.groupby(['Neighborhood']).transform(

    lambda x: x.fillna(x.median())

)

missing=(all_data.isnull().sum().sort_values(ascending=False)/len(all_data))[:35]

missing_data = pd.DataFrame({'Missing Ratio' :missing})

missing_data.head(5)
all_data.head(20)
all_data['MSSubClass']=all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold']=all_data['YrSold'].apply(str)

all_data['MoSold']=all_data['MoSold'].apply(str)
from sklearn.preprocessing import LabelEncoder



cols_for_lab_enc=('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for c in cols_for_lab_enc:

    le=LabelEncoder()

    le.fit(list(all_data[c].values))

    all_data[c] = le.transform(list(all_data[c].values))



all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] +

                                 all_data['1stFlrSF'] + all_data['2ndFlrSF'])



all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +

                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))



all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +

                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +

                              all_data['WoodDeckSF'])
all_data.HouseStyle.unique()
num_feat=all_data.dtypes[all_data.dtypes!="object"].index

num_feat
skews_col=all_data[num_feat].skew().sort_values(ascending=False)

skews_col = skews_col[abs(skews_col) > 0.75]

skews_col
from scipy.special import boxcox1p

from scipy.stats import boxcox

for c in skews_col.index:

    all_data[c]=boxcox1p(all_data[c],0.1)

#     all_data[c]=boxcox(all_data[c].apply(lambda x: x+1 ))[0]
skews_col=all_data[num_feat].skew().sort_values(ascending=False)

skews_col = skews_col[abs(skews_col) > 0.5]

skews_col
# from sklearn.preprocessing import StandardScaler

# for c in num_feat:

#     sc=StandardScaler()

#     all_data[c]=sc.fit_transform(np.array([all_data[c].values]).T)
all_data.head(20)
y=train['SalePrice']

all_data=all_data.drop(['SalePrice'],axis=1)

all_data=pd.get_dummies(all_data)

train=all_data[:len(train)]

test=all_data[len(train):]

# test.head(20)
f, ax = plt.subplots(figsize=(30, 30))

corr2=pd.concat([train,y],axis=1).corr()

sns.heatmap(corr2)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,ElasticNet,Lasso,Ridge

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split,GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR

import xgboost as xgb

import lightgbm as lgb

from bayes_opt import BayesianOptimization

from functools import partial

from sklearn.svm.libsvm import cross_validation

from sklearn.neighbors import KNeighborsRegressor

from mlxtend.regressor import StackingCVRegressor
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2)

kfolds = KFold(n_splits=10, shuffle=True)

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))

def rmsle_cv(model):

    rmse= np.sqrt(-cross_val_score(model, train.values, y.values, scoring="neg_mean_squared_error", cv = kfolds.get_n_splits(train.values)))

    return rmse
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds



    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)



        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

    

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
class AvarageModels(BaseEstimator, TransformerMixin, RegressorMixin):

    def __init__(self,models):

        self.models=models

    def fit(self,X,y):

        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:

            model.fit(X, y)



        return self

    def predict(self,X):

        prediction= np.column_stack([m.predict(X) for m in self.models_])

        return np.mean(prediction,axis=1)
class StackNet(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models,meta_final_model, meta_models1=None, meta_models2=None,add_prev_out=True, n_folds=10):

        self.base_models = base_models

        self.meta_models1 = meta_models1

        self.meta_models2 = meta_models2

        self.meta_final_model=meta_final_model

        self.n_folds = n_folds

        self.add_prev_out=add_prev_out

   

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        if self.meta_models1!=None:

            self.meta_models1_ = [list() for x in self.meta_models1]

        if self.meta_models2!=None:

            self.meta_models2_ = [list() for x in self.meta_models2]

        self.meta_final_model_=clone(self.meta_final_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

       

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

        if self.add_prev_out:

            out_of_fold_predictions=np.hstack((out_of_fold_predictions,X))

        out_of_fold_predictions2 = np.zeros((X.shape[0],len(self.meta_models1)))

       

        if self.meta_models1!=None:

            for i, model in enumerate(self.meta_models1):



                for train_index, holdout_index in kfold.split(out_of_fold_predictions, y):

                    instance = clone(model)

                    self.meta_models1_[i].append(instance)

                    instance.fit(out_of_fold_predictions[train_index], y[train_index])

                    y_pred = instance.predict(out_of_fold_predictions[holdout_index])

                    out_of_fold_predictions2[holdout_index, i] = y_pred                           

            if self.add_prev_out:

                out_of_fold_predictions2=np.hstack((out_of_fold_predictions2,X))

        else:

            out_of_fold_predictions2=out_of_fold_predictions

            out_of_fold_predictions3 = np.zeros((X.shape[0],len(self.meta_models2)))

        

        

        if self.meta_models2!=None:

            for i, model in enumerate(self.meta_models2):



                for train_index, holdout_index in kfold.split(out_of_fold_predictions2, y):

                    instance = clone(model)

                    self.meta_models2_[i].append(instance)

                    instance.fit(out_of_fold_predictions2[train_index], y[train_index])

                    y_pred = instance.predict(out_of_fold_predictions2[holdout_index])

                    out_of_fold_predictions3[holdout_index, i] = y_pred                           

            if self.add_prev_out:

                out_of_fold_predictions3=np.hstack((out_of_fold_predictions3,X))         

        else:

            out_of_fold_predictions3=out_of_fold_predictions2

                                            

        self.meta_final_model_.fit(out_of_fold_predictions3, y)

        return self

   

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_])

        if self.add_prev_out:

            meta_features=np.hstack((meta_features,X))

            

        if self.meta_models1!=None:   

            meta_features2 = np.column_stack([

                np.column_stack([model.predict(meta_features) for model in meta_models1_]).mean(axis=1)

                for meta_models1_ in self.meta_models1_ ])

            if self.add_prev_out:

                meta_features2=np.hstack((meta_features2,X))

        else:

            meta_features2=meta_features

            

        if self.meta_models2!=None:  

            meta_features3 = np.column_stack([

                np.column_stack([model.predict(meta_features2) for model in meta_models2_]).mean(axis=1)

                for meta_models2_ in self.meta_models2_ ])

            if self.add_prev_out:

                meta_features3=np.hstack((meta_features3,X))

        else:

            meta_features3=meta_features2

            

            

        return self.meta_final_model_.predict(meta_features3)
# def fit_with(alpha,l1_ratio,max_iter):

#     enet=make_pipeline(RobustScaler(),ElasticNet(alpha=alpha,l1_ratio=l1_ratio,max_iter=700+int(max_iter*3900)))

#     enet.fit(X_train,y_train)

#     pred=rmsle_cv(enet).mean()

#     return -pred



# pbounds = {'alpha': (0, 0.05),'l1_ratio':(0.7,1),'max_iter':(0,1) }

# optimizer = BayesianOptimization(

#     f=fit_with,

#     pbounds=pbounds,

#     random_state=1,

# )



# optimizer.maximize(init_points=100, n_iter=350, acq="poi", xi=1e-2)

# print(optimizer.max)
max_iter=1e7

l1_ratio=[x/100 for x in range(65,100)]

alphas=[y*(10**-x) for x in range(2,6) for y in range(1,9)]

C=[x for x in range(1,30,3)]

epsilon=[y*(10**-x) for x in range(2,6) for y in range(1,9)]

gamma=[y*(10**-x) for x in range(2,6) for y in range(1,9)]



xgb_1=xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

lgb_1=lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

enet=make_pipeline(RobustScaler(),ElasticNet(alpha=0.0005, l1_ratio=.9))

lasso=make_pipeline(RobustScaler(),Lasso(alpha =0.0005))

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
am = AvarageModels((enet,lasso,GBoost,KRR))

sav=StackingAveragedModels((enet,GBoost,KRR,lasso),lasso)

sn=StackNet((enet,GBoost,KRR,lasso,am,sav),lasso,(KRR,lasso),add_prev_out=False,n_folds=15)
w=rmsle_cv(enet)

print(w.mean(),w.std())

w=rmsle_cv(lasso)

print(w.mean(),w.std())

w=rmsle_cv(GBoost)

print(w.mean(),w.std())

w=rmsle_cv(KRR)

print(w.mean(),w.std())

w=rmsle_cv(am)

print(w.mean(),w.std())

w=rmsle_cv(sav)

print(w.mean(),w.std())

xgb_1.fit(X_train.values,y_train.values)

w=rmsle(xgb_1.predict(X_test.values),y_test.values)

print(w)

lgb_1.fit(X_train.values,y_train.values)

w=rmsle(lgb_1.predict(X_test.values),y_test.values)

print(w)

sn.fit(X_train.values,y_train.values)

w=rmsle(sn.predict(X_test.values),y_test.values)

print(w)
xgb_1.fit(train,y)

lgb_1.fit(train,y)

sn.fit(train.values,y.values)

predicted_prices=np.expm1(sn.predict(test.values))*0.7+np.expm1(xgb_1.predict(test))*0.00+np.expm1(lgb_1.predict(test))*0.3

my_submission = pd.DataFrame({'Id': test_Id, 'SalePrice': predicted_prices})



my_submission.to_csv('submission.csv', index=False)

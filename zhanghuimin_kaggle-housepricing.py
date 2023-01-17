import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from datetime import datetime

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

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



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print ("Data is loaded!")
print ("Train: ",train.shape[0],"sales, and ",train.shape[1],"features")

print ("Test: ",test.shape[0],"sales, and ",test.shape[1],"features")
train.head()
test.head()
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

quantitative.remove('SalePrice')

quantitative.remove('Id')

qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
quantitative
qualitative
sns.set_style("whitegrid")

missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
y = train['SalePrice']

plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=stats.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=stats.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=stats.lognorm)
test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01

normal = pd.DataFrame(train[quantitative])

normal = normal.apply(test_normality)

print(not normal.any())
def encode(frame, feature):

    ordering = pd.DataFrame()

    ordering['val'] = frame[feature].unique()

    ordering.index = ordering.val

    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']

    ordering = ordering.sort_values('spmean')

    ordering['ordering'] = range(1, ordering.shape[0]+1)

    ordering = ordering['ordering'].to_dict()

    

    for cat, o in ordering.items():

        frame.loc[frame[feature] == cat, feature+'_E'] = o

    

qual_encoded = []

for q in qualitative:  

    encode(train, q)

    qual_encoded.append(q+'_E')

print(qual_encoded)
def spearman(frame, features):

    spr = pd.DataFrame()

    spr['feature'] = features

    spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]

    spr = spr.sort_values('spearman')

    plt.figure(figsize=(6, 0.25*len(features)))

    sns.barplot(data=spr, y='feature', x='spearman', orient='h')

    

features = quantitative + qual_encoded

#spearman(train, features)
plt.figure(1)

corr = train[quantitative+['SalePrice']].corr()

sns.heatmap(corr)

plt.figure(2)

corr = train[qual_encoded+['SalePrice']].corr()

sns.heatmap(corr)

plt.figure(3)

corr = pd.DataFrame(np.zeros([len(quantitative)+1, len(qual_encoded)+1]), index=quantitative+['SalePrice'], columns=qual_encoded+['SalePrice'])

for q1 in quantitative+['SalePrice']:

    for q2 in qual_encoded+['SalePrice']:

        corr.loc[q1, q2] = train[q1].corr(train[q2])

sns.heatmap(corr)
features = quantitative + qual_encoded

model = TSNE(n_components=2, random_state=0, perplexity=50)

X = train[features].fillna(0.).values

tsne = model.fit_transform(X)



std = StandardScaler()

s = std.fit_transform(X)

pca = PCA(n_components=30)

pca.fit(s)

pc = pca.transform(s)

kmeans = KMeans(n_clusters=5)

kmeans.fit(pc)



fr = pd.DataFrame({'tsne1': tsne[:,0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})

sns.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)

print(np.sum(pca.explained_variance_ratio_))
train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
train = train[train.GrLivArea < 4500]

train.reset_index(drop=True, inplace=True)

train["SalePrice"] = np.log1p(train["SalePrice"])

y = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)

test_features = test

features = pd.concat([train_features, test_features]).reset_index(drop=True)
features.shape
features['MSSubClass'] = features['MSSubClass'].apply(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)

features['Functional'] = features['Functional'].fillna('Typ') 

features['Electrical'] = features['Electrical'].fillna("SBrkr") 

features['KitchenQual'] = features['KitchenQual'].fillna("TA") 

features["PoolQC"] = features["PoolQC"].fillna("None")

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

features.head()
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)



for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')



features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
features.head()
objects = []

for i in features.columns:

    if features[i].dtype == object:

        objects.append(i)

features.update(features[objects].fillna('None'))



features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

features.update(features[numerics].fillna(0))
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics2.append(i)

skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)



features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
features.shape
final_features = pd.get_dummies(features).reset_index(drop=True)

final_features.shape
X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(y):, :]

X.shape, y.shape, X_sub.shape
outliers = [30, 88, 462, 631, 1322]

X = X.drop(X.index[outliers])

y = y.drop(y.index[outliers])



overfit = []

for i in X.columns:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 99.94:

        overfit.append(i)



overfit = list(overfit)

X = X.drop(overfit, axis=1)

X_sub = X_sub.drop(overfit, axis=1)
X.shape, y.shape, X_sub.shape
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
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

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
score = cv_rmse(ridge)

score = cv_rmse(lasso)

print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(elasticnet)

print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(svr)

print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(lightgbm)

print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(gbr)

print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(xgboost)

print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
def blend_models_predict(X):

    return ((0.1 * elastic_model_full_data.predict(X)) + \

            (0.05 * lasso_model_full_data.predict(X)) + \

            (0.1 * ridge_model_full_data.predict(X)) + \

            (0.1 * svr_model_full_data.predict(X)) + \

            (0.1 * gbr_model_full_data.predict(X)) + \

            (0.15 * xgb_model_full_data.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.3 * stack_gen_model.predict(np.array(X))))
print('RMSLE score on train data:')

print(rmsle(y, blend_models_predict(X)))
print('Predict submission')

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))
print('Blend with Top Kernels submissions\n')

sub_1 = pd.read_csv('../input/top-10-0-10943-stacking-mice-and-brutal-force/House_Prices_submit.csv')

sub_2 = pd.read_csv('../input/hybrid-svm-benchmark-approach-0-11180-lb-top-2/hybrid_solution.csv')

sub_3 = pd.read_csv('../input/lasso-model-for-regression-problem/lasso_sol22_Median.csv')

submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 

                                (0.25 * sub_1.iloc[:,1]) + 

                                (0.25 * sub_2.iloc[:,1]) + 

                                (0.25 * sub_3.iloc[:,1]))
q1 = submission['SalePrice'].quantile(0.005)

q2 = submission['SalePrice'].quantile(0.995)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission.csv", index=False)
submission.head()
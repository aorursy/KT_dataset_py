import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats
%matplotlib inline
train_raw_data = pd.read_csv('../input/train.csv')
test_raw_data = pd.read_csv('../input/test.csv')
train_raw_data.shape
test_raw_data.shape
corr_mat = train_raw_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat, vmax=0.8, square=True)
k = 10
cols = corr_mat.nlargest(k, ['SalePrice'])['SalePrice'].index
cm = np.corrcoef(train_raw_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=cols.values, annot_kws={'size': 10}, 
                 xticklabels=cols.values)
plt.show()
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_raw_data[cols], size=2.5)
plt.show()
n_train = train_raw_data.shape[0]
n_test = test_raw_data.shape[0]
all_data = pd.concat((train_raw_data, test_raw_data), sort=True).reset_index(drop=True)
all_data.drop(['SalePrice'], inplace=True, axis=1)
all_data.shape
def missing_data_stats():
    total = all_data.isnull().sum().sort_values(ascending=False)
    percent = (all_data.isnull().sum() / all_data.shape[0]).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(40))
missing_data_stats()
fill_na_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass']

for col in fill_na_cols:
    all_data[col] = all_data[col].fillna("None")
fill_zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 
                  'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

for col in fill_zero_cols:
    all_data[col] = all_data[col].fillna(0)
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
fill_mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']

for col in fill_mode_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data['Functional'] = all_data['Functional'].fillna('Typ')
missing_data_stats()
train_data = pd.concat((all_data[:n_train], train_raw_data['SalePrice']), axis=1).reset_index(drop=True)
train_data.shape
test_data = all_data[n_train:]
test_data.shape
sale_price_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:, np.newaxis])
low_range = np.sort(sale_price_scaled, axis=0)[:10]
high_range = np.sort(sale_price_scaled, axis=0)[-10:]
print('low range of the distribution')
print(low_range)
print('high range of the distribution')
print(high_range)
var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000))
train_data.sort_values(by = 'GrLivArea', ascending=False)[:2]
train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)
train_data = train_data.drop(train_data[train_data['Id'] == 524].index)

all_data = all_data.drop(all_data[all_data['Id'] == 1299].index)
all_data = all_data.drop(all_data[all_data['Id'] == 524].index)
var = 'TotalBsmtSF'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
def check_normal(dist):
    sns.distplot(dist, fit=norm)
    fig = plt.figure()
    res = stats.probplot(dist, plot=plt)
check_normal(train_data['SalePrice'])
train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
check_normal(train_data['SalePrice'])
check_normal(train_data['GrLivArea'])
train_data['GrLivArea'] = np.log1p(train_data['GrLivArea'])
all_data['GrLivArea'] = np.log1p(all_data['GrLivArea'])
check_normal(train_data['GrLivArea'])
check_normal(train_data['TotalBsmtSF'])
train_data['TotalBsmtSF'] = np.log1p(train_data['TotalBsmtSF'])
all_data['TotalBsmtSF'] = np.log1p(all_data['TotalBsmtSF'])
check_normal(train_data[train_data['TotalBsmtSF'] > 0]['TotalBsmtSF'])
plt.scatter(train_data['GrLivArea'], train_data['SalePrice'])
plt.scatter(train_data[train_data['TotalBsmtSF'] > 0]['TotalBsmtSF'], train_data[train_data['TotalBsmtSF'] > 0]['SalePrice'])
all_data = pd.get_dummies(all_data)
n_train = train_data.shape[0]
x_train_data = all_data[:n_train].values
y_train_data = train_data['SalePrice'].values.reshape(-1, 1)
x_test_data = all_data[n_train:].values
from sklearn.model_selection import KFold, cross_validate, learning_curve, GridSearchCV
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
nfolds = 5
random_seed = 30

# mean squared error using cross validation
def ms_cv(model):
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=random_seed)
    cv = cross_validate(model, x_train_data, y_train_data, cv=kf, scoring='neg_mean_squared_error', return_train_score=True)
    cv['test_score'] = -cv['test_score']
    cv['train_score'] = -cv['train_score']
    return cv['train_score'].mean(), cv['test_score'].mean()
def learn_curve(model):
    train_sizes, train_scores, test_scores = learning_curve(model, x_train_data, y_train_data, scoring='neg_mean_squared_error')
    train_scores_avg = np.mean(train_scores, axis=1)
    test_scores_avg = np.mean(test_scores, axis=1)
    
    fig = plt.figure()
    plt.plot(train_sizes, train_scores_avg, 'b-', label='train data')
    plt.plot(train_sizes, test_scores_avg, 'r-', label='test data')
    plt.yticks(np.linspace(min(test_scores_avg), max(train_scores_avg), 20))
    plt.legend()
    plt.show()
alphas = np.array([])
base = 1e-4
for i in range(10):
    alphas = np.append(alphas, np.linspace(base, base*10, 10))
    base = base*10

lasso_cv = LassoCV(alphas=alphas)
lasso_cv.fit(x_train_data, y_train_data)
lasso_cv.alpha_
lasso_best_model = Lasso(lasso_cv.alpha_)
lasso_ms_cv = ms_cv(lasso_best_model)
print(lasso_ms_cv)
learn_curve(lasso_best_model)
alphas = np.array([])
base = 1e-4
for i in range(10):
    alphas = np.append(alphas, np.linspace(base, base*10, 10))
    base = base*10

elasticnet_cv = ElasticNetCV(alphas=alphas)
elasticnet_cv.fit(x_train_data, y_train_data)
elasticnet_cv.alpha_
elasticnet_best_model = ElasticNet(alpha=0.0009)
elasticnet_ms_cv = ms_cv(elasticnet_best_model)
print(elasticnet_ms_cv)
learn_curve(elasticnet_best_model)
svr = SVR()
svr.get_params()
Cs = np.array([])
base = 1e-4
for i in range(5):
    Cs = np.append(Cs, np.linspace(base, base*10, 3))
    base = base*10
    
params = {'svr__C': Cs, 'svr__epsilon': Cs, 'svr__kernel': ['poly']}
svr_pipeline = make_pipeline(StandardScaler(), SVR(gamma='auto', degree=1))
svr_cv = GridSearchCV(estimator=svr_pipeline, param_grid=params)
svr_cv.fit(x_train_data, y_train_data)
svr_cv.best_params_
svr_pipeline_best_model = svr_cv.best_estimator_
svr_pipeline_ms_cv = ms_cv(svr_pipeline_best_model)
print(svr_pipeline_ms_cv)
learn_curve(svr_pipeline_best_model)
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        for model in self.models_:
            model.fit(X, y)
            
        return self
    
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)
averaged_models = AveragingModels([lasso_best_model, elasticnet_best_model, svr_pipeline_best_model])
print(ms_cv(averaged_models))
learn_curve(averaged_models)
averaged_models.fit(x_train_data, y_train_data)
y_test_data = averaged_models.predict(x_test_data)
y_test_data = np.expm1(y_test_data)
submission = pd.DataFrame()
submission['Id'] = test_raw_data['Id']
submission['SalePrice'] = y_test_data
submission.to_csv('submission.csv', index=False)

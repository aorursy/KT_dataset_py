import pandas as pd



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

ids = test['id']

target = train['price'].values

data = pd.concat([train.drop(['price'], axis=1), test])

print(train.shape, test.shape, data.shape, target.shape)
train.head()
import matplotlib.pyplot as plt



train['price_q'] = pd.qcut(train['price'], q=10, labels=list(range(10))).astype(int)

train['yr_built_q'] = pd.qcut(train['yr_built'], q=10, labels=list(range(10))).astype(int)

train['sqft_living_q'] = pd.qcut(train['sqft_living'], q=10, labels=list(range(10))).astype(int)
def draw_scatter(df, col_name, axes):

    df.plot(kind='scatter', x='long', y='lat', c=col_name, 

            cmap=plt.get_cmap('plasma'), colorbar=False, alpha=0.1, ax=axes)

    axes.set(xlabel='longitude', ylabel='latitude')

    axes.set_title(col_name, fontsize=13)

    return axes

       

fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 9), dpi=100)



draw_scatter(train, 'price_q', axes.flat[0])

draw_scatter(train, 'yr_built_q', axes.flat[1])

draw_scatter(train, 'sqft_living_q', axes.flat[2])

fig.suptitle('Seattle Housing Prices', fontsize=15)

fig.delaxes(axes.flat[3])

fig.tight_layout()

fig.subplots_adjust(top=0.9)

plt.show()
from bokeh.models.mappers import ColorMapper, LinearColorMapper

from bokeh.palettes import Plasma10

from bokeh.plotting import gmap

from bokeh.models import GMapOptions, HoverTool, ColumnDataSource

from bokeh.io import output_notebook, show



output_notebook()



api_key = 'AIzaSyBYrbp34OohAHsX1cub8ZeHlMEFajv15fY'



map_options = GMapOptions(lat=47.5112, lng=-122.257, map_type='roadmap', zoom=10)

p = gmap(api_key, map_options, title='Seattle Housing Prices')



source = ColumnDataSource(

    data=dict(

        lat=train['lat'].tolist(), 

        long=train['long'].tolist(),

        color=train['price_q'].tolist(),

        price=train['price'].tolist()

    )

)



color_mapper = LinearColorMapper(palette=Plasma10)

p.circle(x='long', y='lat', 

         fill_color={'field': 'color', 'transform': color_mapper}, 

         fill_alpha=0.3, line_color=None, source=source)



hover = HoverTool(

    tooltips=[

        ('lat', '@lat'), 

        ('long', '@long'), 

        ('price', '@price')

    ]

)



p.add_tools(hover)

show(p)
print('median: {}'.format(train['price'].median()))

print('mean: {:.3f}'.format(train['price'].mean()))

print('min: {}'.format(train['price'].min()))

print('max: {}'.format(train['price'].max()))

print('std: {:.3f}'.format(train['price'].std()))

print('skewness: {:.3f}'.format(train['price'].skew()))

print('kurtosis: {:.3f}'.format(train['price'].kurt()))
import numpy as np

from scipy import stats

from scipy.stats import norm

plt.style.use('classic')



def draw_plot(log_transform=False, axes=None):

    if log_transform:

        prices = np.log(train['price']+1)

    else:

        prices = train['price']

        

    mu = prices.mean()

    sigma = prices.std()

    

    y = norm.pdf(prices)

    n, bins, patches = axes.flat[0].hist(prices,

                                         bins=60, 

                                         density=True, 

                                         cumulative=False, 

                                         color='blue',

                                         edgecolor='black', 

                                         linewidth=0.3) 



    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

    axes.flat[0].plot(bins, y, '--', color='red')

    axes.flat[0].tick_params(axis='y', left=False, labelleft=False)

    axes.flat[0].tick_params(axis='x', rotation=0)

    axes.flat[0].set_title('Histogram')



    stats.probplot(prices, dist='norm', plot=axes.flat[1])

    axes.flat[1].get_lines()[0].set_marker('o')

    axes.flat[1].get_lines()[0].set_markeredgecolor('black')

    axes.flat[1].get_lines()[0].set_markeredgewidth(0.2)

    axes.flat[1].get_lines()[0].set_markerfacecolor('blue')

    axes.flat[1].get_lines()[1].set_linestyle('--')

    axes.flat[1].get_lines()[1].set_color('red')

    axes.flat[1].tick_params(axis='y', left=True, labelleft=False)

    axes.flat[1].get_yaxis().set_visible(False)

    axes.flat[1].set_title('Probability Plot')

    return axes



fig, axes = plt.subplots(2, 2, figsize=(7,6), dpi=100)

draw_plot(axes=axes.flat[:2])

draw_plot(log_transform=True, axes=axes.flat[2:])

axes.flat[0].tick_params(axis='x', rotation=90)

fig.tight_layout()

plt.show()
import seaborn as sns



fig, axes = plt.subplots(4, 2, sharey=True, figsize=(10, 15), dpi=100)



columns = [

    'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade'

]



for i, col in enumerate(columns):

    train.boxplot(column='price', by=col, ax=axes.flat[i])

    axes.flat[i].set_xlabel('')

    axes.flat[i].set_title(col, fontsize=14)

    axes.flat[i].grid(axis='x')

    

axes.flat[1].tick_params(axis='x', rotation=90)

fig.suptitle('')

fig.delaxes(axes.flat[7])

fig.tight_layout()

plt.show()
fig, axes = plt.subplots(4, 2, figsize=(10, 15), dpi=100)



columns = [

    'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 

    'lat', 'long', 'sqft_living15', 'sqft_lot15'

]



for i, col in enumerate(columns):

    sns.distplot(train[col].values, hist_kws={'alpha': 1}, ax=axes.flat[i])

    axes.flat[i].set_xlabel('')

    axes.flat[i].set_title(col, fontsize=14)

    axes.flat[i].grid(axis='y')

    axes.flat[i].tick_params(axis='x', rotation=90)

    

fig.suptitle('')

# fig.delaxes(axes.flat[7])

fig.tight_layout()

plt.show()
pearson = pd.DataFrame()



columns = [

    'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 

    'lat', 'long', 'sqft_living15', 'sqft_lot15'

]



pearson['features'] = columns

pearson['coefficient'] = [train[col].corr(train['price']) for col in columns]

pearson = pearson.sort_values('coefficient')

pearson.set_index('features', inplace=True)



fig, ax = plt.subplots(figsize=(6, 10), dpi=100)

pearson.plot(kind='barh', ax=ax)

plt.show()
input_data = train.copy()



columns = [

    'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade'

]



for col in columns:

    input_data[col] = input_data[col].astype('category')

    

anova = pd.DataFrame()

anova['features'] = columns

p_values = []



for col in columns:

    samples = []

    for uniq in input_data[col].unique():

        samples.append(input_data[input_data[col] == uniq]['price'].values)

    p = stats.f_oneway(*samples)[1]

    p_values.append(p)



anova['p_values'] = p_values



anova['disparity'] = np.log(1. / anova['p_values'].values+1)



anova.sort_values('disparity', inplace=True, ascending=False)

anova['disparity'] = anova['disparity'].replace({np.inf: 800})



fig = plt.figure(figsize=(5,10))

sns.barplot(data=anova, y='features', x='disparity', color='blue')

plt.xlim([0, 700])

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor



columns = ['sqft_living', 'bathrooms', 'bedrooms', 'lat', 'long', 'price']



X = train[columns].drop(['price'], axis=1)

X['random'] = np.random.random(size=len(X))

y = train['price']

print(X.shape, y.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestRegressor(n_estimators=100, oob_score=True)

rf.fit(X_train, y_train)

default_importances = rf.feature_importances_
from mlxtend.evaluate import feature_importance_permutation



permutation_importances, _ = feature_importance_permutation(

    predict_method=rf.predict, 

    X=X_test.values,

    y=y_test.values,

    metric='r2',

    num_rounds=10, seed=42)
import eli5

from eli5.sklearn import PermutationImportance



eli5_permutation_importances = PermutationImportance(rf, random_state=42).fit(X_test, y_test)

eli5.show_weights(eli5_permutation_importances, feature_names=X_test.columns.tolist())
from sklearn.base import clone



def dropcolumn_importances(estimator, X_train, y_train):

    estimator_ = clone(estimator)

    estimator_.random_state = 42

    estimator_.fit(X_train, y_train)

    baseline = estimator_.oob_score_

    importances = []

    for col in X_train.columns:

        X = X_train.drop(col, axis=1)

        estimator_ = clone(estimator)

        estimator_.random_state = 42

        estimator_.fit(X, y_train)

        o = estimator_.oob_score_

        importances.append(baseline - o)

    importances = np.array(importances)

    return importances
dropcolumn_importances = dropcolumn_importances(rf, X_train, y_train)
def draw_importances(importances, title, ax):

    indices = np.argsort(importances)

    temp = pd.DataFrame({'importances': importances},

                        index=X_train.columns).reset_index()

    temp.sort_values(by='importances', inplace=True)

    temp['colors'] = temp['index'].map(lambda x: False if x == 'random' else True)

    temp['importances'].plot(kind='barh', color=temp['colors'].map({True: 'b', False: 'r'}), ax=ax)

    ax.set_title(title)

    ax.set_yticklabels([X_train.columns[i] for i in indices])
fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)

draw_importances(default_importances, 'Default', axes.flat[0])

draw_importances(permutation_importances, 'Permutation', axes.flat[1])

draw_importances(dropcolumn_importances, 'Drop-column', axes.flat[2])

fig.delaxes(axes.flat[3])

fig.tight_layout()

plt.show()
X = train.drop(['id', 'date', 'price', 'price_q', 'yr_built_q', 'sqft_living_q'], axis=1)

y = np.log(train['price']+1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



rf = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)

rf.fit(X_train, y_train)

print(rf.score(X_train, y_train))

print(rf.score(X_test, y_test))



permutation_importances, _ = feature_importance_permutation(

    predict_method=rf.predict, 

    X=X_test.values,

    y=y_test.values,

    metric='r2',

    num_rounds=10, 

    seed=42)
fig, ax = plt.subplots(figsize=(5, 8), dpi=100)

draw_importances(permutation_importances, 'Permutation Importances', ax)
import statsmodels.api as sm



def draw_outliers(col_name=None, ax=None):

    ols_model = sm.OLS(train['price'], train[col_name]).fit()

    residuals = ols_model.resid

    outlier = train.iloc[np.argmax(residuals.values**2)]    

    sns.regplot(x=col, y='price', data=train, ax=ax, scatter_kws={'color': 'blue', 'edgecolors': 'black'}, line_kws={'color': 'red'})

    ax.scatter(outlier.loc[col], outlier.loc['price'], c='red')



fig, axes = plt.subplots(3, 2, figsize=(10, 15), dpi=100)



columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

for i, col in enumerate(columns):

    draw_outliers(col, ax=axes.flat[i])

    axes.flat[i].set_xlabel('')

    axes.flat[i].set_title(col, fontsize=14)

    axes.flat[i].grid(axis='y')

    axes.flat[i].tick_params(axis='x', rotation=90)

    

fig.suptitle('')

# fig.delaxes(axes.flat[7])

fig.tight_layout()

plt.show()
# outliers = [5108, 6469]

# train.drop(outliers, axis=0, inplace=True)

# target = np.delete(target, outliers)

# print(train.shape, target.shape)
# z = np.abs(stats.zscore(train['sqft_living']))

# print(np.where(z > 3))
train['price_per_sqrt'] = train['price'].div(train['sqft_living'])

price_by_sqrt = train.groupby(by='zipcode')['price_per_sqrt'].agg([('mean_by_sqrt', 'mean'), ('median_by_sqrt', 'median')])

price_by_zipcode = train.groupby(by='zipcode')['price'].agg([('mean_by_zipcode', 'mean'), ('median_by_zipcode', 'median')])

price_by_grade = train.groupby(by='grade')['price'].agg([('mean_by_grade', 'mean'), ('median_by_grade', 'median')])
data = pd.merge(data, price_by_sqrt, left_on='zipcode', right_on='zipcode', )

print(data.shape)



data = pd.merge(data, price_by_zipcode, left_on='zipcode', right_on='zipcode')

print(data.shape)



data = pd.merge(data, price_by_grade, left_on='grade', right_on='grade')

print(data.shape)
data['datetime'] = pd.to_datetime(data['date'].str[:8])

data['date_year'] = data['datetime'].dt.year
data['renovated'] = data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
data.sort_values(by='id', inplace=True)

data.reset_index(drop=True, inplace=True)

data.head()
for col in ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_lot15']:

    data[col] = np.log(data[col]+1)
train = data[:train.shape[0]]

target_log = np.log(target+1)



X = train.drop(['id', 'date', 'datetime'], axis=1)

y = target_log

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



rf = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)

rf.fit(X_train, y_train)

print(rf.score(X_train, y_train))

print(rf.score(X_test, y_test))



permutation_importances, _ = feature_importance_permutation(

    predict_method=rf.predict, 

    X=X_test.values,

    y=y_test,

    metric='r2',

    num_rounds=10, 

    seed=42)
fig, ax = plt.subplots(figsize=(5, 8), dpi=100)

draw_importances(permutation_importances, 'Permutation Importances', ax)

plt.show()
import re

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold





class AverageModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

    

    def fit(self, X, y):

        self.models_ = [clone(x[1]) for x in self.models]

        

        print('Building average models...')

        for i, model in enumerate(self.models_):

            model.fit(X, y)

            print('Finished {} model'.format(i+1))

        return self

    

    def predict(self, X):

        predictions = np.column_stack([model.predict(X) for model in self.models_])

        return np.mean(predictions, axis=1)

    

class StackedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

        

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model[1])

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        print('Building stacked models...')

        

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model[1])

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

            print('Finished {} model'.format(i+1))

            

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

    

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, features):

        self.features = features

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        return X[self.features].values
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline





onehot_features = ['renovated', 'waterfront', 'view']

onehot_pipeline = Pipeline([

    ('selecter', DataFrameSelector(onehot_features)),

    ('cat_encoder', OneHotEncoder(categories='auto'))

])



ordinal_features = ['grade', 'condition']

ordinal_pipeline = Pipeline([

    ('selecter', DataFrameSelector(ordinal_features)),

    ('cat_encoder', OrdinalEncoder(categories='auto'))

])



numeric_features = [

    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',

    'floors', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',

    'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'mean_by_sqrt', 

    'median_by_sqrt', 'mean_by_zipcode', 'median_by_zipcode', 'mean_by_grade',

    'median_by_grade', 'date_year',

]

numeric_pipeline = Pipeline([

    ('selecter', DataFrameSelector(numeric_features)),

    ('cat_encoder', RobustScaler())

])



full_pipeline = FeatureUnion(transformer_list=[

    ('onehot_pipeline', onehot_pipeline), 

    ('ordinal_pipeline', ordinal_pipeline),

    ('numeric_pipeline', numeric_pipeline),

])
from sklearn.linear_model import Ridge

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

import xgboost as xgb

import lightgbm as lgb



params = {

    'alpha': 0.001, 

    'normalize':True,

    'max_iter': 1e7,

    'solver': 'sag',

}



reg1 = make_pipeline(RobustScaler(), Ridge(**params))



params = {

    'kernel': 'rbf', 

    'degree': 3,

    'C': 20,

    'gamma': 'scale',

    'max_iter': -1,

    'epsilon': 0.005,

    'cache_size': 200,

    'shrinking': True,

    'tol': 0.001,

}



reg2 = make_pipeline(RobustScaler(), SVR(**params))



params = {

    'base_estimator': DecisionTreeRegressor(max_depth=6),

    'n_estimators': 200,

    'learning_rate': 0.1,

}



reg3 = AdaBoostRegressor(**params)



params = {

    'max_depth': 9, 

    'n_estimators': 1000,

    'subsample': 0.8,

    'learning_rate': 0.1,

}



reg4 = GradientBoostingRegressor(**params)



params = {

    'max_depth': 8, 

    'n_estimators': 2000,

    'max_features': 'sqrt', 

    'n_jobs': -1, 

}



reg5 = RandomForestRegressor(**params)



params = {

    'objective': 'reg:linear',

    'max_depth': 7, 

    'n_estimators': 5000,

    'reg_alpha': 0.005,

    'reg_lambda': 1,

    'subsample': 0.7,

    'colsample_bytree': 0.8,

    'learning_rate': 0.1,

    'booster ': 'gbtree',

    'n_jobs': -1, 

    'silent': True,

}



reg6 = xgb.XGBRegressor(**params)



params = {

    'objective':'regression',

    'max_depth': -1,

    'n_estimators': 5000,

    'num_leaves': 31,

    'subsample': 0.8,

    'learning_rate': 0.1,

    'min_child_samples': 20,

    'boosting_type': 'gbdt',

    'subsample_freq ': 0.8,

    'reg_alpha': 0.1,

    'n_jobs': -1, 

}



reg7 = lgb.LGBMRegressor(**params)
train = data[:train.shape[0]]

test = data[train.shape[0]:]

target_log = np.log(target+1)



train = train.drop(['id', 'date', 'datetime'], axis=1)

test = test.drop(['id', 'date', 'datetime'], axis=1)

print(train.shape, test.shape, target.shape)
from sklearn.model_selection import cross_val_score



def pipeline_rmsle_cv(model, n_folds=3):

    transformed_train = full_pipeline.fit_transform(train)

    kfold = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(transformed_train)

    rmse = np.sqrt(-cross_val_score(model, transformed_train, np.exp(target_log), scoring='neg_mean_squared_error', cv=kfold, n_jobs=4))

    return rmse



def rmsle_cv(model, n_folds=3):

    kfold = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse = np.sqrt(-cross_val_score(model, train.values, np.exp(target_log), scoring='neg_mean_squared_error', cv=kfold, n_jobs=4))

    return rmse
base_models = [

    ('Ridge', reg1),

    ('AdaBoost', reg3),

    ('Gradient Boosting', reg4),

    ('Random Forest', reg5), 

    ('XGB Regressor', reg6),

    ('LGBM Regressor', reg7),

]



for name, model in base_models:

    score = rmsle_cv(model, n_folds=2)

    print('{} = {:,.3f}({:.3f})\n'.format(name, score.mean(), score.std()))
base_models = [

    ('Ridge', reg1),

    ('AdaBoost', reg3),

    ('Gradient Boosting', reg4),

    ('Random Forest', reg5), 

    ('XGB Regressor', reg6),

    ('LGBM Regressor', reg7),

]



average_models = AverageModels(base_models)

score = rmsle_cv(average_models, n_folds=2)

print('AverageModels = {:,.3f}({:.3f})'.format(score.mean(), score.std()))
base_models = [

    ('AdaBoost', reg3),

    ('Gradient Boosting', reg4),

    ('Random Forest', reg5), 

    ('XGB Regressor', reg6),

    ('LGBM Regressor', reg7),

]



stacked_models = StackedModels(base_models=base_models, meta_model=('Ridge', reg1), n_folds=3)

score = rmsle_cv(stacked_models, n_folds=2)

print('StackedModels = {:,.3f}({:.3f})'.format(score.mean(), score.std()))
# base_models = [

#     ('Gradient Boosting', reg4),

#     ('Random Forest', reg5), 

#     ('XGB Regressor', reg6),

#     ('LGBM Regressor', reg7),

# ]



# average_models = AverageModels(base_models)

# average_models.fit(train, target_log)

# average_models_predition = np.expm1(average_models.predict(test))

# print(average_models_predition[:5])
base_models = [

    ('AdaBoost', reg3),

    ('Gradient Boosting', reg4),

    ('Random Forest', reg5), 

    ('XGB Regressor', reg6),

    ('LGBM Regressor', reg7),

]



stacked_models = StackedModels(base_models=base_models, meta_model=('Ridge', reg1))

stacked_models.fit(train.values, target_log)

stacked_models_predition = np.expm1(stacked_models.predict(test.values))

print(stacked_models_predition[:5])
# preds = (average_models_predition * 0.2) + (stacked_models_predition * 0.8)

preds = stacked_models_predition

submission = pd.DataFrame({'id': ids,

                           'price': preds})



submission.to_csv('./submission.csv', index=False)
# from sklearn.metrics import mean_squared_error



# def rmse(y_true, y_pred):

#     return np.sqrt(mean_squared_error(np.exp(y_true), np.exp(y_pred)))
# function_set = ('add', 'sub', 'mul', 'div', 'sin', 'cos', 'sqrt', 'min', 'max')



# gp = SymbolicTransformer(population_size=2000, 

#                          generations=10,

#                          function_set=function_set,

#                          n_components=10,

#                          parsimony_coefficient=0.0005,

#                          max_samples=0.9, 

#                          verbose=1,

#                          random_state=42)



# gp.fit(X_train, y_train)
# gp_features = gp.transform(X_train)

# X_train_gp = np.hstack((X_train, gp_features))

# print(X_train_gp.shape)
# from sklearn.linear_model import Ridge



# reg = Ridge()

# reg.fit(X_train, y_train)

# y_pred = reg.predict(X_train)

# print(rmse(y_train, y_pred))



# reg = Ridge()

# reg.fit(X_train_gp, y_train)

# y_pred = reg.predict(X_train_gp)

# print(rmse(y_train, y_pred))
# X_new = test.drop(['id', 'date'], axis=1)
# gp_features = gp.transform(X_new)

# X_new_gp = np.hstack((X_new, gp_features))

# print(X_new_gp.shape)
# import pandas as pd

# import numpy as np

# from scipy import stats



# train = pd.read_csv('../input/train.csv')

# test = pd.read_csv('../input/test.csv')

# print(train.shape, test.shape)



# # train = train[(np.abs(stats.zscore(train[cols])) < 5).all(axis=1)].copy()

# # print(train.shape)

# # train.dropna(inplace=True)

# # print(train.shape)



# target = np.log(train['price'].values+1)

# train = train[cols].values
# train = data[:train.shape[0]]

# test = data[train.shape[0]:]

# target = np.log(target+1)

# train = train.drop(['id', 'date', 'datetime'], axis=1).values

# test = test.drop(['id', 'date', 'datetime'], axis=1).values

# print(train.shape, test.shape, target.shape)
# from sklearn.metrics import mean_squared_error



# def rmse(y_true, y_pred):

#     return np.sqrt(mean_squared_error(y_true, y_pred))
# import xgboost as xgb

# from sklearn.model_selection import KFold



# params = {

#     'objective': 'reg:linear',

#     'eval_metric': 'rmse',

#     'max_depth': 10, 

#     'eta': 0.1, 

#     'alpha': 3,

#     'subsample': 0.8,

#     'learning_rates': 0.1,

#     'silent': 1,

# }



# train_preds = np.zeros(len(train))

# xgb_preds = np.zeros(len(test))

# xgb_test = xgb.DMatrix(test)



# kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# for train_index, valid_index in kfold.split(train, y=target):

#     train_data = xgb.DMatrix(train[train_index], label=target[train_index])

#     valid_data = xgb.DMatrix(train[valid_index], label=target[valid_index])

    

#     watch_list = [(train_data, 'train'), (valid_data, 'valid')]

    

#     num_round = 20000

#     bst = xgb.train(params, train_data, num_round, evals=watch_list, 

#                     early_stopping_rounds=500, verbose_eval=False)

    

#     train_preds[valid_index] = bst.predict(valid_data, 

#                                            ntree_limit=bst.best_ntree_limit)

#     xgb_preds += bst.predict(xgb_test, ntree_limit=bst.best_ntree_limit) / 5



# print('RMSE = {:,}'.format(rmse(np.exp(target), np.exp(train_preds))))

# print('RMSE = {:,}'.format(rmse(target, train_preds)))
# import lightgbm as lgb



# params = {

#     'objective':'regression',

#     'metric': 'rmse',

#     'max_depth': -1,

#     'learning_rate': 0.1,

#     'min_child_samples': 16,

#     'boosting': 'gbdt',

#     'feature_fraction': 0.8,

#     'lambda_l1': 0.1,

#     'verbosity': -1,

# }



# train_preds = np.zeros(len(train))

# lgb_preds = np.zeros(len(test))

# lgb_test = lgb.Dataset(test)



# kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# for train_index, valid_index in kfold.split(train, y=target):

#     train_data = lgb.Dataset(train[train_index], label=target[train_index])

#     valid_data = lgb.Dataset(train[valid_index], label=target[valid_index])

    

#     watch_list = [train_data, valid_data]

    

#     num_round = 20000

#     bst = lgb.train(params, train_data, num_round, valid_sets=watch_list, 

#                     early_stopping_rounds=500, verbose_eval=False)

    

#     train_preds[valid_index] = bst.predict(train[valid_index], num_iteration=bst.best_iteration)

#     lgb_preds += bst.predict(test, num_iteration=bst.best_iteration) / 5



# print('RMSE = {:,}'.format(rmse(np.exp(target), np.exp(train_preds))))

# print('RMSE = {:,}'.format(rmse(target, train_preds)))
# fig, ax = plt.subplots(figsize=(50,10), dpi=100)

# xgb.plot_tree(bst, num_trees=42, ax=ax)

# plt.show()
# fig, ax = plt.subplots(figsize=(5,5), dpi=100)

# xgb.plot_importance(bst, ax=ax)

# plt.show()
# dnew = xgb.DMatrix(X_new)

# pred = bst.predict(dnew)

# print(pred[:5])
# preds = (np.exp(xgb_preds) * 0.2) + (np.exp(lgb_preds)) * 0.8

# preds[:5]
# submission = pd.DataFrame({'id': ids,

#                            'price': preds})



# submission.to_csv('./submission.csv', index=False)
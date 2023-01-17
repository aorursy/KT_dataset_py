## Models

from sklearn.linear_model import Ridge, Lasso, ElasticNet, Perceptron, LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

from sklearn.svm import SVR



## Utilities

from sklearn.model_selection import cross_val_score, learning_curve, RandomizedSearchCV

from sklearn.metrics import make_scorer, r2_score, mean_squared_log_error

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



## Core and plotting

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import glob
ts = pd.read_csv('../input/train.csv')

vd = pd.read_csv('../input/test.csv')

df = pd.concat([ts, vd], ignore_index=True)

df['Train'] = df['SalePrice'].notna() * 1
r2 = make_scorer(r2_score)

msle = make_scorer(mean_squared_log_error, greater_is_better=False)
optional_features = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

df.loc[:, optional_features] = df.loc[:, optional_features].fillna('No')
cat_outliers = []

for column in df.select_dtypes('O'):

    counts = df[column].value_counts()

    if len(counts) > 4 or (counts[0] > 5 * counts[1:].sum()):

        cat_outliers.append(column)

        

df.drop(cat_outliers, axis=1, inplace=True)
plt.figure(figsize=(20,4))

plt.scatter(x=df['YearBuilt'], y=df['GarageYrBlt'], c=df['SalePrice'], s=50);

plt.colorbar()

plt.title(u'Correlação')

plt.show();
df.loc[df['GarageYrBlt'] > 2050, 'GarageYrBlt'] = np.nan

df.loc[df['GarageYrBlt'].isna(), 'GarageYrBlt'] = df.loc[df['GarageYrBlt'].isna(), 'YearBuilt'] # Weak Hypothesis (Simplified)
plt.figure(figsize=(20,4))

plt.scatter(x=df['LotFrontage'], y=df['LotArea'], c=df['SalePrice'], s=50);

plt.title(u'Correlação')

plt.colorbar()

plt.show();
df.drop(df[(df['LotFrontage'] > 250)|(df['LotArea']>150000)].index, axis=0, inplace=True)
plt.figure(figsize=(20,4))

df[df['Train']==1]['LotFrontage'].plot(kind='hist', bins=30, alpha=0.6, normed=False, label='Train')

df[df['Train']==0]['LotFrontage'].plot(kind='hist', bins=30, alpha=0.6, normed=False, label='Test')

plt.title("LotFrontAge",  fontsize=20)

plt.legend()

plt.show();
lotfrontage_regression = False

if lotfrontage_regression:

    lotfrontaage_v = ['LotFrontage', 'LotArea','TotalBsmtSF', 'GarageCars', '1stFlrSF', 'TotRmsAbvGrd']

    lotfrontage = df[df['LotFrontage'].notna()][lotfrontaage_v]

    lfa = xgb.XGBRegressor(booster='gbtree')

    lfa.fit(lotfrontage.drop('LotFrontage', axis=1), lotfrontage['LotFrontage'])

    df.loc[df['LotFrontage'].isna(), 'LotFrontage'] = lfa.predict(df[df['LotFrontage'].isna()][lotfrontaage_v].drop('LotFrontage', axis=1))

else:

    df.loc[df['LotFrontage'].isna(), 'LotFrontage'] = df[df['LotFrontage'].notna()]['LotFrontage'].mean()
plt.figure(figsize=(10,4))

df[df['Train']==1]['OverallQual'].plot(kind='hist', bins=10, alpha=0.5, normed=False, label='Train')

df[df['Train']==0]['OverallQual'].plot(kind='hist', bins=10, alpha=0.5, normed=False, label='Test')

plt.title("OverallQual",  fontsize=20)

plt.legend()

plt.show();
plt.figure(figsize=(10,4))

df[df['Train']==1]['OverallCond'].plot(kind='hist', bins=9, alpha=0.5, normed=False, label='Train')

df[df['Train']==0]['OverallCond'].plot(kind='hist', bins=9, alpha=0.5, normed=False, label='Test')

plt.title("OverallCond",  fontsize=20)

plt.legend()

plt.show();
plt.figure(figsize=(10,4))

df[df['Train']==1]['GarageCars'].plot(kind='hist', bins=5, alpha=0.5, normed=False, label='Train')

df[df['Train']==0]['GarageCars'].plot(kind='hist', bins=6, alpha=0.5, normed=False, label='Test')

plt.title("GarageCars",  fontsize=20)

plt.legend()

plt.show();
plt.figure(figsize=(20,4))

plt.scatter(df['TotalBsmtSF'], df['1stFlrSF'], c=df['SalePrice'], cmap='RdBu')

plt.colorbar()

print("# Insight: new feature 'has basement?'")
plt.figure(figsize=(20,4))

plt.scatter(df['GrLivArea'], df['TotRmsAbvGrd'], c=df['Train'], cmap='RdBu');
plt.figure(figsize=(30,10))

sns.heatmap(df.corr(method='spearman'), cmap='RdBu');
df.drop(['Utilities', 'Train'], axis=1, inplace=True)
for col in df.columns:

    if df[col].dtype in ['int64', 'float64'] and not col in ['Id', 'SalePrice']:

        df.loc[df[col].isna(), col] = df.loc[df[col].notna(), col].median()

        df[col] = df[col].astype('float64')

    elif df[col].dtype == 'object':

        df.loc[df[col].isna(), col] = df[col].value_counts().index[0]

        df[col] = df[col].astype('category')



df['Id'] = df['Id'].astype('str')

df['SalePrice'] = df['SalePrice'].apply(np.round)
plt.subplots(1, 2, figsize=(20,4))

plt.subplot(1, 2, 1)

df['SalePrice'].plot(kind='hist', bins=30)

plt.subplot(1, 2, 2)

df['SalePrice'] = df['SalePrice'].apply(np.log)

df['SalePrice'].plot(kind='hist', bins=30)

plt.show();
df.describe(include='all')
df['HasBasement'] = df['TotalBsmtSF'] > 0
df.plot.scatter('YearBuilt', 'YearRemodAdd');

print('''# Insight: new feature 'Was remodeled?'

# Insight: new feature 'Was remodeled on build year?\'''')



df['Remodeled'] = df['YearRemodAdd'] > 1950

df['RemodeledOnBltYr'] = df['YearRemodAdd'] != df['YearBuilt']
df.index = df['Id']
df = pd.get_dummies(df)

train, test = df.loc[df['SalePrice'].notna(), :], df.loc[df['SalePrice'].isna(), :]

x_train, y_train = train.drop('SalePrice', axis=1), train['SalePrice']

x_test , y_test  =  test.drop('SalePrice', axis=1),  test['SalePrice']
TRAIN_TEST_PROP = df['SalePrice'].isna().astype(int).mean()

print("Proportion of test set: %.1f%%"%(TRAIN_TEST_PROP*100))
def mycv(nfolds=5):

    for _ in range(nfolds):

        yield train_test_split(shuffle(np.arange(len(y_train))), test_size=TRAIN_TEST_PROP)
model_search = False

if model_search:

    N_CROSS_VAL = 5

    models = [ExtraTreeRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), Perceptron(), XGBRegressor(), SVR(), Ridge()]



    def model_name(model):

        return str(model).split('(')[0]

    results = pd.DataFrame(index=map(model_name, models), columns=['mean', 'std'] + ["#%d"%(x+1) for x in range(N_CROSS_VAL)])

    for model in tqdm.tqdm(models):

        values = cross_val_score(model, x_train, y_train, scoring='neg_mean_squared_log_error', cv=mycv(N_CROSS_VAL), n_jobs=-1)

        print(values)

        results.loc[model_name(model), :] = [values.mean(), values.std()] + list(values)

    results = results.sort_values(by='mean', ascending=True)

    results['model'] = results.index

    results = pd.melt(results, value_vars=['#1', '#2', '#3', '#4', '#5'], id_vars=['model'])
if model_search:

    plt.figure(figsize=(20,4))

    sns.boxplot(x=results['value'].astype('float'), y=results['model'], showfliers=False)

    plt.show()
x_train.isnull().any().any() or y_train.isnull().any()
reg = XGBRegressor()

params = {

    'learning_rate': np.logspace(-2, 0, 5),

    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],

    'reg_alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],

    'reg_lambda': [0.001, 0.01, 0.1, 1],

    'n_estimators': [50, 100, 200, 300, 1000],

}

tune_model = False

if tune_model:

    rscv = RandomizedSearchCV(reg, params, 20, 'neg_mean_squared_error', n_jobs=-1, cv=mycv(3), return_train_score=False, verbose=True)

    rscv.fit(x_train, y_train)

    results = pd.DataFrame(rscv.cv_results_).sort_values(by='mean_test_score', ascending=True)[['mean_test_score', 'std_test_score', 'param_max_depth', 'param_learning_rate', 'param_reg_alpha', 'param_reg_lambda', 'param_n_estimators']]

    display(results.head(10))

    model = rscv.best_estimator_

else:

    reg.fit(x_train, y_train)

    model = reg
final_params = {

    'mean_test_score': -15444966768.470783,

     'std_test_score': 399396003.6882163,

     'param_max_depth': 7,

     'param_learning_rate': 0.01,

     'param_reg_alpha': 0.005,

     'param_reg_lambda': 0.1,

     'param_n_estimators': 50

}



if not tune_model:

    final_params = results.iloc[0].to_dict()
print_learning_curve = True

if print_learning_curve:

    plt.figure(figsize=(20,5))

    n, ts, vd = learning_curve(model, x_train, y_train, scoring='neg_mean_squared_log_error', cv=mycv(2), train_sizes=np.linspace(0.1, 1.0, 4))

    plt.plot(n, ts.mean(axis=1), color='b', label='Training')

    plt.plot(n, vd.mean(axis=1), color='r', label='Validation')

    plt.title(u"Learning Curve")

    plt.legend(fontsize=20)

    plt.show();
test['SalePrice'] = np.exp(model.predict(x_test))
plt.subplots(1, 2, figsize=(20,4))

plt.subplot(1, 2, 1)

np.exp(df['SalePrice']).plot(kind='hist', bins=30)

plt.subplot(1, 2, 2)

test['SalePrice'].plot(kind='hist', bins=30)

plt.show();
test[['SalePrice']].reset_index().to_csv('output.csv', index=None)
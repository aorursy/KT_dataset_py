# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.width', 1000)

import scipy.stats as stats



# Plotting Tools

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns



# Import Sci-Kit Learn

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold



# Ensemble Models

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



# Package for stacking models

from vecstack import stacking



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from IPython.display import display, HTML

display(HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

</style>

"""))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def show_all(df):

    #This fuction lets us view the full dataframe

    with pd.option_context('display.max_rows', 100, 'display.max_columns', 100):

        display(df)
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')



show_all(train.head())
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
f, ax = plt.subplots(figsize=(16, 8))

sns.lineplot(x='OverallQual', y='SalePrice', data=train)
f, ax = plt.subplots(figsize=(16, 8))

sns.lineplot(x='GrLivArea', y='SalePrice', data=train)
f, ax = plt.subplots(figsize=(16, 8))

sns.lineplot(x='GarageArea', y='SalePrice', data=train)
f, ax = plt.subplots(figsize=(16, 8))

sns.lineplot(x='GarageCars', y='SalePrice', data=train)
f, ax = plt.subplots(figsize=(16, 8))

sns.lineplot(x='TotalBsmtSF', y='SalePrice', data=train)
f, ax = plt.subplots(figsize=(16, 8))

sns.lineplot(x='1stFlrSF', y='SalePrice', data=train)
f, ax = plt.subplots(figsize=(16, 8))

sns.lineplot(x='FullBath', y='SalePrice', data=train)
f, ax = plt.subplots(figsize=(16, 8))

sns.lineplot(x='TotRmsAbvGrd', y='SalePrice', data=train)
f, ax = plt.subplots(figsize=(16, 8))

sns.lineplot(x='YearBuilt', y='SalePrice', data=train)
plt.figure(1); plt.title('Johnson SU')

sns.distplot(train['SalePrice'], kde=False, fit=stats.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(train['SalePrice'], kde=False, fit=stats.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(train['SalePrice'], kde=False, fit=stats.lognorm)
sns.catplot(x='SaleType', y='SalePrice', data=train, kind='bar', palette='muted')
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], height = 2.5)

plt.show();
# Find and plot categories in train that are missing data

sns.set_style("whitegrid")

missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
missing
# Fix missing data

def fill_missing_values(df):

    ''' This function imputes missing values with median for numeric columns 

        and most frequent value for categorical columns'''

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    for column in list(missing.index):

        if df[column].dtype == 'object':

            df[column].fillna(df[column].value_counts().index[0], inplace=True)

        elif df[column].dtype == 'int64' or 'float64' or 'int16' or 'float16':

            df[column].fillna(df[column].median(), inplace=True)
fill_missing_values(train)

train.isnull().sum().max()
# Find and plot categories in train that are missing data

sns.set_style("whitegrid")

missing = test.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
fill_missing_values(test)

test.isnull().sum().max()
list(train.select_dtypes(exclude=[np.number]).columns)
def impute_cats(df):

    '''This function converts categorical and non-numeric 

       columns into numeric columns to feed into a ML algorithm'''

    # Find the columns of object type along with their column index

    object_cols = list(df.select_dtypes(exclude=[np.number]).columns) # Create a list from the dataframe that includes only non-numeric columns

    object_cols_ind = []                                              # Create a list

    for col in object_cols:

        object_cols_ind.append(df.columns.get_loc(col))               # Get index of the column within the datafram



    # Encode the categorical columns with numbers    

    label_enc = LabelEncoder()                                        # Encodes labels with value between 0 and n_classes - 1

    for i in object_cols_ind:

        df.iloc[:,i] = label_enc.fit_transform(df.iloc[:,i])
# Impute the missing values

impute_cats(train)

impute_cats(test)

print("Train Dtype counts: \n{}".format(train.dtypes.value_counts()))

print("Test Dtype counts: \n{}".format(test.dtypes.value_counts()))
def rmse(y, y_pred):

    return np.sqrt(mean_squared_error(np.log(y), np.log(y_pred)))
X = train.drop('SalePrice', axis=1)

y = np.ravel(np.array(train[['SalePrice']]))

print(y.shape)
# Use train_test_split from sci-kit learn to segment our data into train and a local testset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Initialize the model

random_forest = RandomForestRegressor(n_estimators=1200,

                                      max_depth=15,

                                      min_samples_split=5,

                                      min_samples_leaf=5,

                                      max_features=None,

                                      random_state=42,

                                      oob_score=True

                                     )



# Perform cross-validation to see how well our model does 

kf = KFold(n_splits=5)

y_pred = cross_val_score(random_forest, X, y, cv=kf, n_jobs=-1)

y_pred.mean()
def rmse(y, y_pred):

    return np.sqrt(mean_squared_error(np.log(y), np.log(y_pred)))
# Initialize the model

random_forest = RandomForestRegressor(n_estimators=1200,

                                      max_depth=15,

                                      min_samples_split=5,

                                      min_samples_leaf=5,

                                      max_features=None,

                                      random_state=42,

                                      oob_score=True

                                     )



# Perform cross-validation to see how well our model does 

kf = KFold(n_splits=5)

y_pred = cross_val_score(random_forest, X, y, cv=kf, n_jobs=-1)

y_pred.mean()
# Fit the model to our data

random_forest.fit(X, y)
# Make predictions on test data

rf_pred = random_forest.predict(test)
# Initialize our model

xg_boost = XGBRegressor( learning_rate=0.01,

                         n_estimators=6000,

                         max_depth=4, min_child_weight=1,

                         gamma=0.6, subsample=0.7,

                         colsample_bytree=0.2,

                         objective='reg:linear', nthread=-1,

                         scale_pos_weight=1, seed=27,

                         reg_alpha=0.00006

                       )



# Perform cross-validation to see how well our model does 

kf = KFold(n_splits=5)

y_pred = cross_val_score(xg_boost, X, y, cv=kf, n_jobs=-1)

y_pred.mean()
# Fit our model to the training data

xg_boost.fit(X,y)
# Make predictions on the test data

xgb_pred = xg_boost.predict(test)
# Initialize our model

g_boost = GradientBoostingRegressor( n_estimators=6000, learning_rate=0.01,

                                     max_depth=5, max_features='sqrt',

                                     min_samples_leaf=15, min_samples_split=10,

                                     loss='ls', random_state =42

                                   )



# Perform cross-validation to see how well our model does 

kf = KFold(n_splits=5)

y_pred = cross_val_score(g_boost, X, y, cv=kf, n_jobs=-1)

y_pred.mean()
# Fit our model to the training data

g_boost.fit(X,y)
# Make predictions on test data

gbm_pred = g_boost.predict(test)
# Initialize our model

lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=6,

                                       learning_rate=0.01, 

                                       n_estimators=6400,

                                       verbose=-1,

                                       bagging_fraction=0.80,

                                       bagging_freq=4, 

                                       bagging_seed=6,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                    )



# Perform cross-validation to see how well our model does

kf = KFold(n_splits=5)

y_pred = cross_val_score(lightgbm, X, y, cv=kf)

print(y_pred.mean())
# Fit our model to the training data

lightgbm.fit(X,y)
# Make predictions on test data

lgb_pred = lightgbm.predict(test)
# List of the models to be stacked

models = [g_boost, xg_boost, lightgbm, random_forest]
# Perform Stacking

S_train, S_test = stacking(models,

                           X_train, y_train, X_test,

                           regression=True,

                           mode='oof_pred_bag',

                           metric=rmse,

                           n_folds=5,

                           random_state=25,

                           verbose=2

                          )
# Initialize 2nd level model

xgb_lev2 = XGBRegressor(learning_rate=0.1, 

                        n_estimators=500,

                        max_depth=3,

                        n_jobs=-1,

                        random_state=17

                       )



# Fit the 2nd level model on the output of level 1

xgb_lev2.fit(S_train, y_train)
# Make predictions on the localized test set

stacked_pred = xgb_lev2.predict(S_test)

print("RMSE of Stacked Model: {}".format(rmse(y_test,stacked_pred)))

y1_pred_L1 = models[0].predict(test)

y2_pred_L1 = models[1].predict(test)

y3_pred_L1 = models[2].predict(test)

y4_pred_L1 = models[3].predict(test)

S_test_L1 = np.c_[y1_pred_L1, y2_pred_L1, y3_pred_L1, y4_pred_L1]
test_stacked_pred = xgb_lev2.predict(S_test_L1)
# Save the predictions in form of a dataframe

submission = pd.DataFrame()



submission['Id'] = np.array(test.index)

submission['SalePrice'] = test_stacked_pred
sub1 = pd.read_csv('../input/house-prices-ensemble-7models/House_price_submission.csv')

sub2 = pd.read_csv('../input/1-house-prices-solution-top-1/best_submission.csv')
final_blend = (0.4*sub1.SalePrice.values + 0.4*sub2.SalePrice.values + 0.2*test_stacked_pred)



blended_submission = pd.DataFrame()



blended_submission['Id'] = np.array(test.index)

blended_submission['SalePrice'] = final_blend
submission.to_csv('submission.csv', index=False)

blended_submission.to_csv('blended_submission.csv', index=False) # Best LB Score
from IPython.display import FileLink

FileLink('blended_submission.csv')
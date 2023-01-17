import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import os
#First, taking a look at the list files that is available available
print(os.listdir("../input/"))
application_train_df = pd.read_csv('../input/application_train.csv')
application_test_df = pd.read_csv('../input/application_test.csv')
application_train_df.head()
application_test_df.head()
application_train_df.shape
application_test_df.shape
#A look at their descriptive statistics
application_train_df.describe()
application_test_df.describe()
application_train_df.info()
#Showing null details (missing values) of the application_train_df
application_train_df.isnull().sum()
plt.figure(figsize = (10,5))
sns.countplot(x='TARGET', data = application_train_df)
plt.figure(figsize = (10,5))
sns.countplot(x='NAME_CONTRACT_TYPE', data = application_train_df, hue='TARGET')
plt.figure(figsize = (10,5))
sns.countplot(x='CODE_GENDER', data = application_train_df, hue='TARGET')
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(x='FLAG_OWN_CAR', data = application_train_df, order=('Y','N'),hue='TARGET')
plt.subplot(1,2,2)
sns.countplot(x='FLAG_OWN_REALTY', data = application_train_df, hue='TARGET')
plt.subplots_adjust(wspace = 0.8)
ax = sns.factorplot(x="NAME_INCOME_TYPE", hue="NAME_CONTRACT_TYPE", col="TARGET", data=application_train_df, kind="count",size=4, aspect=1.2)
ax.set_xticklabels(rotation=90)
ax = sns.factorplot(hue="NAME_INCOME_TYPE", x="CODE_GENDER", 
                    col="TARGET",data=application_train_df, 
                    kind="count",size=4, aspect=1.2)
ax = sns.factorplot(x='OCCUPATION_TYPE', data = application_train_df,
                    hue = 'TARGET', kind = 'count', size=4, aspect=1.8)
ax.set_xticklabels(rotation=90)
plt.figure(figsize = (10,5))
sns.countplot(x='NAME_TYPE_SUITE', data = application_train_df)
sns.factorplot(x="AMT_INCOME_TOTAL", y="NAME_INCOME_TYPE", 
               hue="CODE_GENDER", data=application_train_df, size=4, 
                aspect=2.5, kind="bar", ci=False)
sns.factorplot(x="AMT_INCOME_TOTAL", y="OCCUPATION_TYPE", 
               data=application_train_df[application_train_df.OCCUPATION_TYPE.notnull()],
               size=4, aspect=2.5, kind="bar", ci=False)
plt.figure(figsize = (12,5))
sns.countplot(x='NAME_EDUCATION_TYPE', data = application_train_df)
plt.figure(figsize = (12,5))
sns.countplot(x='NAME_HOUSING_TYPE', data = application_train_df)
ax = sns.factorplot(x='ORGANIZATION_TYPE', data = application_train_df, kind = 'count', size=8, aspect=1.8)
ax.set_xticklabels(rotation=90)
plt.figure(figsize=(10,5))
sns.distplot(application_train_df['DAYS_BIRTH'], bins = 50, color = 'red')
plt.title('Showing the Age Distribution')
sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
sns.distplot(application_train_df['AMT_CREDIT'], bins = 30, kde = False, color = 'red')
plt.title('Showing the Credit Amount Distribution')
sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
sns.distplot(application_train_df['AMT_ANNUITY'].dropna(), bins = 30, color = 'red')
plt.title('Showing the Annuity Amount Distribution')
sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
sns.distplot(application_train_df['AMT_GOODS_PRICE'].dropna(), bins = 30, color = 'red')
plt.title('Showing the Goods Price Amount Distribution')
sns.set_style('whitegrid')
sns.jointplot('AMT_CREDIT','AMT_GOODS_PRICE', data=application_train_df, size=8)
sns.lmplot(x='AMT_CREDIT', y='AMT_GOODS_PRICE', data=application_train_df, 
           hue='TARGET', size =7 , col='NAME_CONTRACT_TYPE')
sns.lmplot(x='AMT_CREDIT', y='AMT_GOODS_PRICE', data=application_train_df,
           hue='NAME_CONTRACT_TYPE', size = 7, col='TARGET')
#A quick visual of the entire dataframe showing missing values. It can be seen that much of the apartment and 
# own car age data are missing.
plt.figure(figsize = (10,5))
sns.heatmap(application_train_df.isnull(), yticklabels=False, cbar = False, cmap = 'viridis')
application_train_df['DAYS_EMPLOYED'].plot.hist()
plt.xlabel('DAYS_EMPLOYED')
anomalous_days_employed_train = application_train_df[application_train_df['DAYS_EMPLOYED'] == 365243]
print(len(anomalous_days_employed_train))
#Replacing the outliers with NaNs
application_train_df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
application_train_df['DAYS_EMPLOYED'].plot.hist()
plt.xlabel('DAYS_EMPLOYED')
def impute_suite(NAME_TYPE_SUITE):
    if pd.isnull(NAME_TYPE_SUITE):
        return 'Unaccompanied'
    else:
        return NAME_TYPE_SUITE
application_train_df['NAME_TYPE_SUITE'] = application_train_df['NAME_TYPE_SUITE'].apply(impute_suite)
def impute_occupation(cols):
    OCCUPATION_TYPE=cols[0]
    AMT_INCOME_TOTAL=cols[1]
    if pd.isnull(OCCUPATION_TYPE):
        if AMT_INCOME_TOTAL > 150000.000:
            return 'Laborers'
        elif AMT_INCOME_TOTAL < 150000.000:
            return 'Sales staff'
    else:
        return OCCUPATION_TYPE
application_train_df['OCCUPATION_TYPE'] = application_train_df[['OCCUPATION_TYPE', 'AMT_INCOME_TOTAL']].apply(impute_occupation, axis=1)
def impute_fond(FONDKAPREMONT_MODE):
    if pd.isnull(FONDKAPREMONT_MODE):
        return 'reg oper account'
    else:
        return FONDKAPREMONT_MODE
application_train_df['FONDKAPREMONT_MODE'] = application_train_df['FONDKAPREMONT_MODE'].apply(impute_fond)
def impute_housetype(HOUSETYPE_MODE):
    if pd.isnull(HOUSETYPE_MODE):
        return 'block of flats'
    else:
        return HOUSETYPE_MODE
application_train_df['HOUSETYPE_MODE'] = application_train_df['HOUSETYPE_MODE'].apply(impute_housetype)
application_train_df['WALLSMATERIAL_MODE'] = application_train_df['WALLSMATERIAL_MODE'].fillna(application_train_df['WALLSMATERIAL_MODE'].value_counts().index[0])
application_train_df['EMERGENCYSTATE_MODE'] = application_train_df['EMERGENCYSTATE_MODE'].fillna(application_train_df['EMERGENCYSTATE_MODE'].value_counts().index[0])
#Imputing NaN values in numeric columns with column median. This action completes the clean up.
application_train_df = application_train_df.fillna(application_train_df.median())
#Cleaned dataframe.
plt.figure(figsize = (10,5))
sns.heatmap(application_train_df.isnull(), yticklabels=False, cbar = False, cmap = 'viridis')
corr = application_train_df.corr

plt.figure(figsize = (12,10))
sns.heatmap(corr(), annot = True)
application_train_df = pd.get_dummies(application_train_df, prefix_sep='_', drop_first=True)
print('Below is our newly transformed application_train_df now ready to be trained')
print('Application_train_df shape: ', application_train_df.shape)
application_train_df.head()
#Showing null details (missing values) of the application_test_df
application_test_df.isnull().sum()
application_test_df['DAYS_EMPLOYED'].plot.hist()
plt.xlabel('DAYS_EMPLOYED')
anomalous_days_employed_test = application_test_df[application_test_df['DAYS_EMPLOYED'] == 365243]
print(len(anomalous_days_employed_test))
#Replacing the outliers with NaNs
application_test_df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
application_test_df['DAYS_EMPLOYED'].plot.hist()
plt.xlabel('DAYS_EMPLOYED')
def impute_suite(NAME_TYPE_SUITE):
    if pd.isnull(NAME_TYPE_SUITE):
        return 'Unaccompanied'
    else:
        return NAME_TYPE_SUITE
application_test_df['NAME_TYPE_SUITE'] = application_test_df['NAME_TYPE_SUITE'].apply(impute_suite)
def impute_occupation(cols):
    OCCUPATION_TYPE=cols[0]
    AMT_INCOME_TOTAL=cols[1]
    if pd.isnull(OCCUPATION_TYPE):
        if AMT_INCOME_TOTAL > 150000.000:
            return 'Laborers'
        elif AMT_INCOME_TOTAL < 150000.000:
            return 'Sales staff'
    else:
        return OCCUPATION_TYPE
application_test_df['OCCUPATION_TYPE'] = application_test_df[['OCCUPATION_TYPE', 'AMT_INCOME_TOTAL']].apply(impute_occupation, axis=1)
def impute_fond(FONDKAPREMONT_MODE):
    if pd.isnull(FONDKAPREMONT_MODE):
        return 'reg oper account'
    else:
        return FONDKAPREMONT_MODE
application_test_df['FONDKAPREMONT_MODE'] = application_test_df['FONDKAPREMONT_MODE'].apply(impute_fond)
def impute_housetype(HOUSETYPE_MODE):
    if pd.isnull(HOUSETYPE_MODE):
        return 'block of flats'
    else:
        return HOUSETYPE_MODE
application_test_df['HOUSETYPE_MODE'] = application_test_df['HOUSETYPE_MODE'].apply(impute_housetype)
application_test_df['WALLSMATERIAL_MODE'] = application_test_df['WALLSMATERIAL_MODE'].fillna(application_test_df['WALLSMATERIAL_MODE'].value_counts().index[0])
application_test_df['EMERGENCYSTATE_MODE'] = application_test_df['EMERGENCYSTATE_MODE'].fillna(application_test_df['EMERGENCYSTATE_MODE'].value_counts().index[0])
application_test_df = application_test_df.fillna(application_test_df.median())
plt.figure(figsize = (10,5))
sns.heatmap(application_test_df.isnull(), yticklabels=False, cbar = False, cmap = 'viridis')
application_test_df = pd.get_dummies(application_test_df, prefix_sep='_', drop_first=True)
print('Below is our newly transformed application_test_df')
print('Application_test_df shape: ', application_test_df.shape)
application_test_df.head()
print('Training Features shape: ', application_train_df.shape)
print('Testing Features shape: ', application_test_df.shape)
#Saving the Target column to re-add it again later
application_train_df_TARGET = application_train_df['TARGET']
application_train_df.drop('TARGET', axis=1, inplace = True)
application_train_df, application_test_df = application_train_df.align(application_test_df, join = 'inner', axis = 1)
#ALigned dataframes
print('Training Features shape: ', application_train_df.shape)
print('Testing Features shape: ', application_test_df.shape)
application_train_df = pd.concat([application_train_df, application_train_df_TARGET], axis=1)
application_train_df.head()
print('Training Features shape: ', application_train_df.shape)
print('Testing Features shape: ', application_test_df.shape)
X = application_train_df.drop('TARGET', axis = 1)
y = application_train_df['TARGET']

X_test = application_test_df #This will be called to test our trained and validated model.
from sklearn.model_selection import train_test_split 
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=1001)
import lightgbm as lgbm
#Creating our Training and Validation Sets respectively
lgbm_train = lgbm.Dataset(data=x_train, label=y_train)
lgbm_valid = lgbm.Dataset(data=x_valid, label=y_valid)
#We now define our parameters
params = {}
params['task'] = 'train'
params['objective'] = 'binary'
params['boosting_type'] = 'gbdt'
params['num_iteration'] = 10000
params['learning_rate'] = 0.003
params['metric'] = 'auc'
params['num_leaves'] = 80
params['min_data_in_leaf'] = 100
params['max_depth'] = 8
params['min_child_weight'] = 80
params['reg_alpha'] = 0.05
params['reg_lambda'] = 0.08
params['min_split_gain'] = 0.03
params['sub_sample'] = 0.9
params['colsample_bytree'] = 0.95
lgbm_model = lgbm.train(params, lgbm_train, valid_sets=lgbm_valid, early_stopping_rounds=200, verbose_eval=200)
#Limiting the number of features to displayed to 120
lgbm.plot_importance(lgbm_model, figsize=(12, 25), max_num_features=120)
proba_predictions = lgbm_model.predict(X_test)
submit_lgbm = pd.DataFrame()
submit_lgbm['SK_ID_CURR'] = application_test_df['SK_ID_CURR']
submit_lgbm['TARGET'] = proba_predictions
submit_lgbm.to_csv("lgbm_baseline_model.csv", index=False)
submit_lgbm.head(20)

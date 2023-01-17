import numpy as np

import math

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
print('Dataframe shape:', df.shape)
df.info()
df.head()
def null_percentage(data):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)

    return pd.concat([total, percent], axis=1, keys=['Total','Percent Missing'])



print('''Null values:\n

{}'''.format(null_percentage(df)))
cols = df.columns

colours = ['#000099', '#ffff00']

sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'RISK_MM'], axis=1, inplace=True)
numerical_cols = [var for var in df.columns if df[var].dtype=='f8']

categorical_cols = [var for var in df.columns if df[var].dtype=='O']
print('Numerical Columns: \n{}\n'.format(numerical_cols))

print('Categorical Columns: \n{}\n'.format(categorical_cols))
df.describe()
for var in categorical_cols:

    print(var, ' has {} unique values'.format(len(df[var].unique())))
df['Date'] = pd.to_datetime(df['Date'])



# Extracting Year, Month and Day from Date Column

df['Year'] = df['Date'].dt.year

df['Month'] = df['Date'].dt.month

df['Day'] = df['Date'].dt.day



# Dropping original Date column

df.drop('Date', inplace=True, axis=1)
# Reviewing date changes

df.head()
categorical_cols = [var for var in df.columns if df[var].dtype=='O']



categorical_nulls = df[categorical_cols].isnull().sum()



for var in categorical_cols:

    print(var, ' has: \n{} unique values\n {} null values\n'.format(len(df[var].unique()), categorical_nulls[var]))
def location_percentage(data):

    total = df['Location'].value_counts()

    percent = round(df['Location'].value_counts()/len(df)*100,2)

    return pd.concat([total, percent], axis=1, keys=['Total','%'])



print('''Location Values:

{}'''.format(location_percentage(df)))
# Location dummies

location_dummies = pd.get_dummies(df.Location, drop_first=True).head()

location_dummies.head()
def WindGustDir_percentage(data):

    total = df['WindGustDir'].value_counts()

    percent = round(df['WindGustDir'].value_counts()/len(df)*100,2)

    return pd.concat([total, percent], axis=1, keys=['Total','%'])



WindGustDir_null = df['WindGustDir'].isnull().sum() / len(df['WindGustDir'])



print('''WindGustDir Values:

{}

Null percentage = {}%'''.format(WindGustDir_percentage(df), round(WindGustDir_null * 100, 2)))
# WidGustDir Dummies

WindGustDir_dummies = pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True)

WindGustDir_dummies.head()
def WindDir9am_percentage(data):

    total = df['WindDir9am'].value_counts()

    percent = round(df['WindDir9am'].value_counts()/len(df)*100,2)

    return pd.concat([total, percent], axis=1, keys=['Total','%'])



WindDir9am_null = df['WindDir9am'].isnull().sum() / len(df['WindDir9am'])



print('''WindDir9am Values:

{}

Null percentage = {}%'''.format(WindDir9am_percentage(df), round(WindDir9am_null * 100, 2)))
# WindDir9am Dummies

WindDir9am_dummies = pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True)

WindDir9am_dummies.head()
def WindDir3pm_percentage(data):

    total = df['WindDir3pm'].value_counts()

    percent = round(df['WindDir3pm'].value_counts()/len(df)*100,2)

    return pd.concat([total, percent], axis=1, keys=['Total','%'])



WindDir3pm_null = df['WindDir3pm'].isnull().sum() / len(df['WindDir3pm'])



print('''WindDir3pm Values:

{}

Null percentage = {}%'''.format(WindDir3pm_percentage(df), round(WindDir3pm_null * 100, 2)))
# WindDir3pm Dummies

WindDir3pm_dummies = pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True)

WindDir3pm_dummies.head()
def RainToday_percentage(data):

    total = df['RainToday'].value_counts()

    percent = round(df['RainToday'].value_counts()/len(df)*100,2)

    return pd.concat([total, percent], axis=1, keys=['Total','%'])



RainToday_null = df['RainToday'].isnull().sum() / len(df['RainToday'])



print('''RainToday Values:

{}

Null percentage = {}%'''.format(RainToday_percentage(df), round(RainToday_null * 100, 2)))
# RainToday Dummies

RainToday_dummies = pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True)

RainToday_dummies.head()
# Exploring 'RainTomorrow' values (labels)

def rain_tomorrow_percentage(data):

    total = df['RainTomorrow'].value_counts()

    percent = round(df['RainTomorrow'].value_counts()/len(df)*100,2)

    return pd.concat([total, percent], axis=1, keys=['Total','%'])



print('''RainTomorrow Values:

{}'''.format(rain_tomorrow_percentage(df)))
f, ax2 = plt.subplots(figsize=(5, 5))

ax2 = sns.countplot(x="RainTomorrow", data=df, palette='Blues')

plt.show()
df[numerical_cols].head()
round(df[numerical_cols].describe(), 2)
# As we can see from the difference in 75% percentile and max values, it is likely we have outliers in:

# 'Rainfall', 'WindSpeed9am' and 'WindSpeed3pm'



# Plotting suspected outliers

plt.figure(figsize=(15,10))



plt.subplot(3, 1, 2)

fig = sns.boxplot(x='Rainfall', data=df)

fig.set_title('')



plt.subplot(3, 2, 2)

fig = sns.boxplot(x='WindSpeed9am', data=df)

fig.set_title('')



plt.subplot(3, 2, 1)

fig = sns.boxplot(x='WindSpeed3pm', data=df)

fig.set_title('')
# Plotting Histograms to check skew



plt.figure(figsize=(15,10))



plt.subplot(3, 1, 2)

fig = df['Rainfall'].hist(bins=20)

fig.set_xlabel('Rainfall')



plt.subplot(3, 2, 2)

fig = df['WindSpeed9am'].hist(bins=20)

fig.set_xlabel('WindSpeed9am')



plt.subplot(3, 2, 1)

fig = df['WindSpeed3pm'].hist(bins=20)

fig.set_xlabel('WindSpeed3pm')
#IQR for Rainfall

Q1 = df['Rainfall'].quantile(0.25)

Q3 = df['Rainfall'].quantile(0.75)

IQR = Q3 - Q1

Lower_bound = Q1 - (IQR * 1.5)

Upper_bound = Q3 + (IQR * 1.5)

print('Rainfall has outliers: < {} or > {}'.format(Lower_bound, Upper_bound))
# Removing Rainfall outliers

df = df[~((df['Rainfall'] < - 1.20) |(df['Rainfall'] > 2.0))]

print(df.shape)
#IQR for WindSpeed9am

Q1 = df['WindSpeed9am'].quantile(0.25)

Q3 = df['WindSpeed9am'].quantile(0.75)

IQR = Q3 - Q1

Lower_bound = Q1 - (IQR * 1.5)

Upper_bound = Q3 + (IQR * 1.5)

print('WindSpeed9am has outliers: < {} or > {}'.format(Lower_bound, Upper_bound))
# Removing WindSpeed9am outliers

df = df[~((df['WindSpeed9am'] < - 11.0) |(df['WindSpeed9am'] > 37.0))]

print(df.shape)
#IQR for WindSpeed3pm

Q1 = df['WindSpeed3pm'].quantile(0.25)

Q3 = df['WindSpeed3pm'].quantile(0.75)

IQR = Q3 - Q1

Lower_bound = Q1 - (IQR * 1.5)

Upper_bound = Q3 + (IQR * 1.5)

print('WindSpeed3pm has outliers: < {} or > {}'.format(Lower_bound, Upper_bound))
# Removing WindSpeed3pm outliers

df = df[~((df['WindSpeed3pm'] < - 3.5) |(df['WindSpeed3pm'] > 40.5))]

print(df.shape)
# Reviewing Histograms after outlier removal

plt.figure(figsize=(15,10))



plt.subplot(3, 1, 2)

fig = df['Rainfall'].hist(bins=20)

fig.set_xlabel('Rainfall')



plt.subplot(3, 2, 2)

fig = df['WindSpeed9am'].hist(bins=20)

fig.set_xlabel('WindSpeed9am')



plt.subplot(3, 2, 1)

fig = df['WindSpeed3pm'].hist(bins=20)

fig.set_xlabel('WindSpeed3pm')
# Viewing number of nulls 

pd.DataFrame(df[numerical_cols].isnull().sum().sort_values(ascending=False)).head(12)
# Filling null numericals with mean

for col in numerical_cols:

    mean = df[col].mean()

    df[col].fillna(mean, inplace=True)  
corr_matrix = df.corr()



plt.figure(figsize=(16,12))

plt.title('Correlation Heatmap of Rain in Australia Dataset')

ax = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='white')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           

plt.show()
categorical_cols
df = pd.concat([df, pd.get_dummies(df['Location'],dummy_na=True, prefix='Location', columns=categorical_cols)],axis=1).drop(['Location'],axis=1)

df = pd.concat([df, pd.get_dummies(df['WindGustDir'],dummy_na=True, prefix='WindGustDir', columns=categorical_cols)],axis=1).drop(['WindGustDir'],axis=1)

df = pd.concat([df, pd.get_dummies(df['WindDir9am'],dummy_na=True, prefix='WindDir9am', columns=categorical_cols)],axis=1).drop(['WindDir9am'],axis=1)

df = pd.concat([df, pd.get_dummies(df['WindDir3pm'],dummy_na=True, prefix='WindDir3pm', columns=categorical_cols)],axis=1).drop(['WindDir3pm'],axis=1)

df = pd.concat([df, pd.get_dummies(df['RainToday'],dummy_na=True, prefix='RainToday', columns=categorical_cols)],axis=1).drop(['RainToday'],axis=1)

df.head()
X = df.drop('RainTomorrow', axis=1)

y = df['RainTomorrow']



print('''X Shape: {}

y Shape: {}'''.format(X.shape, pd.DataFrame(y).shape))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



# Checking shapes of each set

print('''X train: {}

X test: {}

y train: {}

y test: {}'''.format(X_train.shape, X_test.shape, pd.DataFrame(y_train).shape, pd.DataFrame(y_test).shape))
scaler = StandardScaler()

cols = pd.DataFrame(X_train).columns



X_train = pd.DataFrame(scaler.fit_transform(X_train), columns= cols)

X_test = pd.DataFrame(scaler.transform(X_test), columns=cols)
# Viewing scaled training set

X_train.head()
# Viewing scaled test set

X_test.head()
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score



random_state = 42
log_reg = LogisticRegression(random_state=random_state)

log_reg.fit(X_train, y_train)



log_reg_pred = log_reg.predict(X_test)



log_reg_cm = confusion_matrix(y_test, log_reg_pred)



print('Model accuracy score:\n{}\nConfusion Matrix:\n{}'. format(round(accuracy_score(y_test, log_reg_pred), 4), log_reg_cm))
# Checking for over/under fitting

print('Training set score: \n{}\nTest set score: \n{}'.format(round(log_reg.score(X_train, y_train), 4), round(log_reg.score(X_test, y_test), 4)))
param_grid = {'C' : [1, 25, 50, 75, 100]}



log_reg_2 = LogisticRegression(random_state=random_state, solver='lbfgs')



grid_search_log = GridSearchCV(log_reg_2, param_grid, scoring="roc_auc", cv=5)



grid_search_log.fit(X_train, y_train)
print('Best Parameters:\n{}'.format(grid_search_log.best_params_))
# Using our best parameters

log_reg_3 = LogisticRegression(random_state=random_state, C=50)

log_reg_3.fit(X_train, y_train)



log_reg_3_pred = log_reg_3.predict(X_test)



log_reg_3_cm = confusion_matrix(y_test, log_reg_3_pred)



print('Model accuracy score:\n{}\nConfusion Matrix:\n{}'. format(round(accuracy_score(y_test, log_reg_3_pred), 4), log_reg_3_cm))
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)



xgb_cm = confusion_matrix(y_test, xgb_pred)



print('Model accuracy score:\n{}\nConfusion Matrix:\n{}'. format(round(accuracy_score(y_test, xgb_pred), 4), xgb_cm))
# Checking for over/under fitting

print('Training set score: \n{}\nTest set score: \n{}'.format(round(xgb.score(X_train, y_train), 4), round(xgb.score(X_test, y_test), 4)))
param_grid = {

     'eta'    : [0.01, 0.15, 0.30 ] ,

     'max_depth'        : [ 3, 6, 9],

     'min_child_weight' : [ 1, 3, 5],

     'gamma'            : [ 0.0, 0.2, 0.4 ]

     }



xgb_2 = XGBClassifier(random_state=random_state)



grid_search = GridSearchCV(xgb_2, param_grid, n_jobs=4, scoring="roc_auc", cv=5)



grid_search.fit(X_train, y_train)
print('Best Parameters:\n{}'.format(grid_search.best_params_))
# Using our best parameters

xgb_3 = XGBClassifier(eta=0.01, gamma=0.4, max_depth=9, min_child_weight=3)

xgb_3.fit(X_train, y_train)



xgb_3_pred = xgb_3.predict(X_test)



xgb_3_cm = confusion_matrix(y_test, xgb_3_pred)



print('Model accuracy score:\n{}\nConfusion Matrix:\n{}'. format(round(accuracy_score(y_test, xgb_3_pred), 4), xgb_3_cm))
# Accuracy scores of each model

log_reg_acc = round(accuracy_score(y_test, log_reg_pred), 4)

log_reg_3_acc = round(accuracy_score(y_test, log_reg_3_pred), 4)

xgb_acc = round(accuracy_score(y_test, xgb_pred), 4)

xgb_3_acc = round(accuracy_score(y_test, xgb_3_pred), 4)
# Creating dataframe showing accuracy scores of each model

compare = {'Model': ['Logistic Regression Original', 'Logistic Regression Tuned', 'XGBoost Original', 'XGBoost Tuned'],

          'Accuracy score': [log_reg_acc, log_reg_3_acc, xgb_acc, xgb_3_acc]}



pd.DataFrame(data=compare)

 
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing other libraries that are required for our study

import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('bmh') #Setting matplot style option to 'Bayesian Methods for Hackers style'

#setting max number of columns to display == 100 in pandas options.
pd.options.display.max_columns = 100
#train set
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

#test set
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#train_df
print(train_df.shape)

train_df.head()
#test_df
print(test_df.shape)

test_df.head()
df_train_test = pd.concat([train_df, test_df], axis=0, join='outer', ignore_index=False, keys=None,
          levels=None, names=None, verify_integrity=False, copy=True) 
print(df_train_test.shape)

df_train_test.head(10)
df_train_test.info()
(df_train_test.isnull().sum()/len(df_train_test) * 100).round(2)
#No of columns with atleast 1 NaN Values in both train and test sets together
print('No of columns with atleast 1 NaN Values:',
      (df_train_test.isnull().sum()/len(df_train_test) * 100).round(2)[(df_train_test.isnull().sum()/len(df_train_test) * 100).round(2) > 0.00].count())
#The Features with 30% and more NaN values 
(df_train_test.isnull().sum()/len(df_train_test) * 100).round(2)[(df_train_test.isnull().sum()/len(df_train_test) * 100).round(2) >= 30.00]
#list of Features that can be droped for now
cols_with_30pct_n_more = (df_train_test.isnull().sum()/len(df_train_test) * 100).round(2)[(df_train_test.isnull().sum()/len(df_train_test) * 100).round(2) >= 30.00].index.to_list()

cols_with_30pct_n_more.remove('SalePrice') #As we are not going to deal with missing values in this

cols_with_30pct_n_more
#Lets also drop Id column alogng with the above
cols_to_drop = ['Id']

#adding the columns to be dropped due to high NaN values
cols_to_drop.extend(cols_with_30pct_n_more)

cols_to_drop
#dropping

df_train_test = df_train_test.drop(cols_to_drop, axis = 1)
#Checking the shape of df after dropping 

df_train_test.shape
#No of columns with atleast 1 NaN Values in both train and test sets together 
#After dropping top cols of them
print('No of columns with atleast 1 NaN Values:',
      (df_train_test.isnull().sum()/len(df_train_test) * 100).round(2)[(df_train_test.isnull().sum()/len(df_train_test) * 100).round(2) > 0.00].count())
print('Columns with atleast 1 NaN Values:',
      (df_train_test.isnull().sum()/len(df_train_test) * 100).round(2)[(df_train_test.isnull().sum()/len(df_train_test) * 100).round(2) > 0.00].sort_values(ascending = False))
nan_cols = (df_train_test.isnull().sum()/len(df_train_test) * 100).round(2)[(df_train_test.isnull().sum()/len(df_train_test) * 100).round(2) > 0.00].sort_values(ascending = False).index.to_list()

nan_cols
#their data types

t_f_obj_nan_cols = (df_train_test[nan_cols].dtypes == object)
t_f_obj_nan_cols
#Type casting bool to str
t_f_obj_nan_cols = t_f_obj_nan_cols.astype('str')

t_f_obj_nan_cols
#list object type features
obj_nan_cols = t_f_obj_nan_cols[t_f_obj_nan_cols == 'True'].index.to_list()

#list of numeric type features
num_nan_cols = t_f_obj_nan_cols[t_f_obj_nan_cols == 'False'].index.to_list()

print("object type features:")
print(obj_nan_cols)
print('\n')
print("numeric type features:")
print(num_nan_cols)
from sklearn.preprocessing import LabelEncoder

for i in range(df_train_test.shape[1]):
    if df_train_test.iloc[:,i].dtypes == object:
        lbl = LabelEncoder()
        lbl.fit(list(df_train_test.iloc[:,i].values))
        df_train_test.iloc[:,i] = lbl.transform(list(df_train_test.iloc[:,i].values))

print(df_train_test['SaleCondition'].unique())
#Removing SalePrice from num_nan_cols
num_nan_cols.remove('SalePrice')
#imputing NaNs

df_train_test[num_nan_cols] = df_train_test[num_nan_cols].fillna(df_train_test[num_nan_cols].median())

#df_train_test[obj_nan_cols] = df_train_test[obj_nan_cols].fillna(df_train_test[obj_nan_cols].mode())
for column in df_train_test[obj_nan_cols]:
    mode = df_train_test[column].mode()
    df_train_test[column] = df_train_test[column].fillna(mode)
print(df_train_test.isnull().sum()[df_train_test.isnull().sum()>0])
df_train_test['TotalSFA'] = df_train_test['TotalBsmtSF'] + df_train_test['1stFlrSF'] + df_train_test['2ndFlrSF']

df_train_test[['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'TotalSFA']].head()
df_train_test[df_train_test.SalePrice.isnull() == True].shape
df_test = df_train_test[df_train_test.SalePrice.isnull() == True]

df_test = df_test.drop('SalePrice', axis = 1)

df_test.columns
df_train = df_train_test.dropna(axis = 0)
df_train.shape
df = df_train
#Target variable -> SalePrice

df.SalePrice.head()
print(df.SalePrice.describe().round(2))
plt.figure(figsize=(9, 8))
sns.distplot(df.SalePrice, color='orange', bins=100, hist_kws={'alpha': 0.4}); #; will avoid the matplotlib verbose informations
# log-transform the target variable for normality
df['SalePrice'] = np.log(df['SalePrice'])

plt.figure(figsize=(9, 8))
sns.distplot(df.SalePrice, color='orange', bins=100, hist_kws={'alpha': 0.4});
df.head()
df.dtypes.unique()
df.describe()
df.hist(figsize = (30, 35), bins = 50, xlabelsize = 8, ylabelsize = 8, color='orange');
df_corr = df.corr()

#Only the reltion coefficients between all other features to SalePrice.
df_corr = df_corr.SalePrice 

df_corr = df_corr.drop('SalePrice')# Because we dont need the correlation SalePrice - SalePrice
#strong correlation
#sorted in descending order of correlation
strong_corr_features = df_corr[abs(df_corr) > 0.6].sort_values(ascending = False) #abs() to avoid the effect of sign

print('There are {} strongly correlated features with SalePrice:\n{}'.format(len(strong_corr_features), strong_corr_features))
corr = df.drop('SalePrice', axis=1).corr() # We already examined SalePrice correlations
plt.figure(figsize=(25, 25))

sns.heatmap(corr[(corr >= 0.8) | (corr <= -0.8)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
#Spliting data to X_train, y_train and X_test
y_train = df['SalePrice']

X_train = df.drop('SalePrice', axis = 1)

X_test = df_test
# feature importance using random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=80, max_features='auto')
rf.fit(X_train, y_train)
print('Training done using Random Forest')

ranking = np.argsort(-rf.feature_importances_)
f, ax = plt.subplots(figsize=(18, 12))
sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')
ax.set_xlabel("Feature Importance")
plt.tight_layout()
plt.show()
# use the top 30 features only
X_train = X_train.iloc[:,ranking[:30]]
X_test = X_test.iloc[:,ranking[:30]]

# interaction between the top 2
X_train["Interaction"] = X_train["TotalSFA"]*X_train["OverallQual"]
X_test["Interaction"] = X_test["TotalSFA"]*X_test["OverallQual"]

# zscoring
X_train = (X_train - X_train.mean())/X_train.std()
X_test = (X_test - X_test.mean())/X_test.std()
    
# heatmap
f, ax = plt.subplots(figsize=(11, 5))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.heatmap(X_train, cmap=cmap)
plt.show()
# relation to the target
fig = plt.figure(figsize=(12,7))
for i in np.arange(30):
    ax = fig.add_subplot(5,6,i+1)
    sns.regplot(x=X_train.iloc[:,i], y=y_train)

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(12,7))
sns.regplot(x=X_train.iloc[:,3], y=y_train)
plt.show()
fig = plt.figure(figsize=(12,7))
sns.regplot(x=X_train.iloc[:,7], y=y_train)
plt.show()
X_temp = X_train
X_temp['SalePrice'] = y_train
X_temp = X_temp.drop(X_temp[(X_temp['TotalSFA']>5) & (X_temp['SalePrice']<12.5)].index)
X_temp = X_temp.drop(X_temp[(X_temp['GrLivArea']>5) & (X_temp['SalePrice']<13)].index)
X_temp = X_temp.drop(X_temp[(X_temp['GarageArea']>3) & (X_temp['SalePrice']<12.5)].index)
X_temp = X_temp.drop(X_temp[(X_temp['BsmtFinSF1']>2) & (X_temp['SalePrice']>13.25)].index)
X_temp = X_temp.drop(X_temp[(X_temp['BsmtFinSF1']>-1) & (X_temp['SalePrice']<11.2)].index)
# recover
y_train = X_temp['SalePrice']
X_train = X_temp.drop(['SalePrice'], axis=1)
# XGBoost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

print("Parameter optimization")
xgb_model = xgb.XGBRegressor()
reg_xgb = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
reg_xgb.fit(X_train, y_train)
print(reg_xgb.best_score_)
print(reg_xgb.best_params_)
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

model = KerasRegressor(build_fn=create_model, verbose=0)
# define the grid search parameters
optimizer = ['SGD','Adam']
batch_size = [10, 30, 50]
epochs = [10, 50, 100]
param_grid = dict(optimizer=optimizer, batch_size=batch_size, epochs=epochs)
reg_dl = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
reg_dl.fit(X_train, y_train)

print(reg_dl.best_score_)
print(reg_dl.best_params_)
# SVR
from sklearn.svm import SVR

reg_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
reg_svr.fit(X_train, y_train)

print(reg_svr.best_score_)
print(reg_svr.best_params_)
# second feature matrix
X_train2 = pd.DataFrame( {'XGB': reg_xgb.predict(X_train),
     'NN': reg_dl.predict(X_train).ravel(),
     'SVR': reg_svr.predict(X_train),
    })
X_train2.head()
# second-feature modeling using linear regression
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(X_train2, y_train)

# prediction using the test set
X_test2 = pd.DataFrame( {'XGB': reg_xgb.predict(X_test),
     'DL': reg_dl.predict(X_test).ravel(),
     'SVR': reg_svr.predict(X_test),
    })

# Don't forget to convert the prediction back to non-log scale
y_pred = np.exp(reg.predict(X_test2))
#y_pred = np.exp(reg_xgb.predict(X_test))
y_pred
test_Id = test_df['Id']
submission = pd.DataFrame({ 
    "Id": test_Id, 
    "SalePrice": y_pred }) 

submission.to_csv('houseprice_111.csv', index=False)

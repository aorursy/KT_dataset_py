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
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv") 

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv") 
train.shape , test.shape
train.head()
test.head()
train.info()
test.info()
int_col=train.loc[:, train.dtypes == np.int64].columns.tolist()
train[int_col]=train[int_col].astype(np.int32)
int_col=test.loc[:, test.dtypes == np.int64].columns.tolist()
test[int_col]=test[int_col].astype(np.int32)
df=pd.concat([train.iloc[:,:-1],test],axis=0).drop(columns=['Id'],axis=1)
df.head()
df.shape
for column in df.columns:    

    if df[column].dtype  == 'object':

        df[column].fillna(value = 'None', inplace=True)

    else:

        df[column].fillna(value = df[column].mean(), inplace=True)
df.columns
import seaborn as sns

sns.distplot(train.SalePrice)
train["SalePrice"] = np.log1p(train["SalePrice"])
import seaborn as sns

sns.distplot(train.SalePrice)
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric = []

for i in df.columns:

    if df[i].dtype in numeric_dtypes:

        numeric.append(i)
from matplotlib import pyplot as plt

sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 7))

ax.set_xscale("log")

ax = sns.boxplot(data=df[numeric] , orient="h", palette="Set1")

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
skew_features = df[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))

skewness = pd.DataFrame({'Skew' :high_skew})

skew_features.head(10)
from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

for i in skew_index:

    df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1))
from matplotlib import pyplot as plt

sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 7))

ax.set_xscale("log")

ax = sns.boxplot(data=df[numeric] , orient="h", palette="Set1")

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
df['Total_Home_Quality'] = df['OverallQual'] + df['OverallCond']

df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
X=pd.get_dummies(df.iloc[:1460,:]).values

y=train.SalePrice.values
X.shape, y.shape
from sklearn.model_selection import train_test_split

X_trains,X_test,y_trains,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
import lightgbm as lgb

from lightgbm import LGBMRegressor

lightgbm = LGBMRegressor(objective='regression', 

                       num_leaves=6,

                       learning_rate=0.01, 

                       n_estimators=7000,

                       max_bin=200, 

                       bagging_fraction=0.8,

                       bagging_freq=4, 

                       bagging_seed=8,

                       feature_fraction=0.2,

                       feature_fraction_seed=8,

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1,

                       metric= 'rmse',

                       random_state=42)

lightgbm.fit(X_trains,y_trains)
y_pred=lightgbm.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

print('mean_squared_error: ',mean_squared_error(y_test, y_pred),

     '\nr2_score: ',r2_score(y_test, y_pred))
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_trains,y_trains)
y_pred = reg.predict(X_test)
print('mean_squared_error: ',mean_squared_error(y_test, y_pred),

     '\nr2_score: ',r2_score(y_test, y_pred))
"""from sklearn.model_selection import GridSearchCV

gridsearch_params = [

    (max_depth, min_child_weight,n_estimators)

    for max_depth in range(9,12)

    for min_child_weight in range(5,8)

    for n_estimators in range(1200,1500,100)

]"""
from xgboost import XGBRegressor

xgboost = XGBRegressor(learning_rate=0.01,

                       n_estimators=6000,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:linear',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       eval_metric='rmse',

                       reg_alpha=0.00006,

                       random_state=42)
xgboost.fit(X_trains,y_trains)
y_pred = xgboost.predict(X_test)
print('mean_squared_error: ',mean_squared_error(y_test, y_pred),

     '\nr2_score: ',r2_score(y_test, y_pred))
"""gridsearch_params = [

    (max_depth, min_child_weight,n_estimators,min_samples_split,min_samples_leaf)

    for max_depth in range(10,15)

    for n_estimators in range(800,1500,100)

    for min_samples_split in range(3,7)

    for min_samples_leaf in range(3,7)

                         

]"""
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=1200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True,

                          random_state=42)
rf.fit(X_trains,y_trains)
y_pred = rf.predict(X_test)
print('mean_squared_error: ',mean_squared_error(y_test, y_pred),

     '\nr2_score: ',r2_score(y_test, y_pred))
X_topred=pd.get_dummies(df.iloc[:1459,:]).values
pred=lightgbm.predict(X_topred)
test.Id.shape
pred.shape
pred
pred=np.expm1(pred)
pred
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": pred

    })
submission
submission.to_csv('predict.csv',index=False)
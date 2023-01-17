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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style='whitegrid')
import cufflinks as cf
cf.go_offline()
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import os
from plotly.subplots import make_subplots
from plotly import tools 
import plotly.graph_objects as go
from scipy import stats


sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train.columns
train.info
test.columns
test.info()
train.head()
test.head()
train.describe()
sns.stripplot(data=train.SalePrice, jitter=True)

train.drop(train[train.SalePrice > 510000].index, inplace=True)
from scipy.stats import norm

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

sns.distplot(train.SalePrice, fit=norm, ax=ax1)
sns.distplot(np.log1p(train.SalePrice), fit=norm, ax=ax2)
sns.distplot(train['SalePrice']);
sns.distplot(np.log(train["SalePrice"]))
sns.scatterplot(x='YrSold', y='SalePrice', data=train)
train.shape
test.shape
train_null = pd.DataFrame(train.isnull().sum().sort_values(ascending = False))
train_null.columns = ['Null']
train_null
train  = train.fillna(train.mean())
test_null=  pd.DataFrame(test.isnull().sum().sort_values(ascending = False))
test_null.columns = ['Null']
test_null
test = test.fillna(test.mean())
X = df.drop(['Id','SalePrice'], axis=1)
y = np.log(train.SalePrice)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(X_train,y_train)
print('Score:', model.score(X_test,y_test))
from sklearn.metrics import mean_squared_error
predic = model.predict(X_test)
print('Rmse:',mean_squared_error(y_test,predic))
plt.scatter(predic, y_test, alpha=.75, color='darkblue')
plt.xlabel('predicted price')
plt.ylabel('actual sale price ')
plt.title('Linear regression ')
plt.show()
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 500, random_state = 0)
rf.fit(X_train, y_train)
rf_pred= rf.predict(X_test)
rf_pred = rf_pred.reshape(-1,1)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, X_train=X_train, y_train=y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
test_model(svr_reg)
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=21)
test_model(dt_reg)
import xgboost
xgb_reg = xgboost.XGBRegressor(bbooster='gbtree', random_state=51)
test_model(xgb_reg)
xgb2_reg=xgboost.XGBRegressor(n_estimators= 899,
 mon_child_weight= 2,
 max_depth= 4,
 learning_rate= 0.05,
 booster= 'gbtree')

test_model(xgb2_reg)
xgb2_reg.fit(X_train,y_train)
y_pred = np.exp(xgb2_reg.predict(X_test))
submit_test = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test.columns=['Id', 'SalePrice']
submit_test.to_csv('sample_submission.csv', index=False)
submit_test
svr_reg.fit(X_train,y_train)
y_pred = np.exp(svr_reg.predict(X_test))
submit_test = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test.columns=['Id', 'SalePrice']
submit_test.to_csv('sample_submission.csv', index=False)
submit_test

corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
k = 15
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.75)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

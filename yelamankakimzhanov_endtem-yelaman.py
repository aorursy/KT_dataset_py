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
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew 
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
print('Shape of train:',train.shape)
print('Shape of test:',test.shape)
train.head()

test.head()

sns.scatterplot(x='YrSold', y='SalePrice', data=train)
train_null = pd.DataFrame(train.isnull().sum().sort_values(ascending = False))
train_null.columns = ['Null']
train_null
train  = train.fillna(train.mean())
test_null=  pd.DataFrame(test.isnull().sum().sort_values(ascending = False))
test_null.columns = ['Null']
test_null
test = test.fillna(test.mean())
df = train.select_dtypes(include=[np.number]).interpolate().dropna()
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
import xgboost
xgb = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
xgb.fit(X_train,y_train)
test_features = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
preds = xgb.predict(test_features)
final_preds = np.exp(preds)

sample_submission['SalePrice'] = final_preds
#final submission  
sample_submission.to_csv('test_submit.csv', index=False)
xgb.fit(X_train,y_train)
test_features = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
preds = xgb.predict(test_features)
final_preds = np.exp(preds)

sample_submission = pd.DataFrame()
sample_submission['ID'] = test['Id']
sample_submission['SalePrice'] = final_preds
#final submission  
sample_submission.to_csv('Pricepred.csv', index=False)




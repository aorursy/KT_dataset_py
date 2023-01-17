# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import sklearn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Store data
##dic = pd.read_excel('../input/Data_Dictionary.xlsx')
##mer = pd.read_csv('../input/merchants.csv')
##tran = pd.read_csv('../input/new_merchant_transactions.csv')
##his = pd.read_csv('../input/historical_transactions.csv')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.describe()
train.head()
# Feature_1 and 2 as they're categorical so use one-hot encoding to convert
frames = pd.get_dummies(train['feature_1'],prefix='feature_1',drop_first=True)
train = pd.concat([ train, frames ],axis=1)
frames = pd.get_dummies(train['feature_2'],prefix='feature_2',drop_first=True)
train = pd.concat([ train, frames ],axis=1)
train = train.drop(['feature_1','feature_2'],axis=1)
# Plot distribution of y in train
plt.hist(train['target'],bins=50)
# Build pairplot and correlation matrix to look at relationship between variables
sns.pairplot(train)
train.corr()
# Store X and y train
df = train
y = train['target']
X = train[['feature_3','feature_1_2','feature_1_3','feature_1_4','feature_1_5','feature_2_2','feature_2_3']]
# Fit RF regressor model to train
rf = RandomForestRegressor(max_features='sqrt')
rf.fit(X,y)
# Showing the most important x variables/features in the model
feature_importances = pd.DataFrame(
    rf.feature_importances_,
    index = X.columns,
    columns = [ 'y' ] 
).sort_values( 'y', ascending = False )

print( feature_importances )
# Determine the accuracy of the random forest applied to train
y_predicted = rf.predict(X)

# Calculate RMSE using the sqrt of MSE
mse = sklearn.metrics.mean_squared_error(y,y_predicted)
rmse = np.sqrt(mse)
rmse
# Prepare test
# Feature_1 and 2 as they're categorical so use one-hot encoding to convert
frames = pd.get_dummies(test['feature_1'],prefix='feature_1',drop_first=True)
test = pd.concat([ test, frames ],axis=1)
frames = pd.get_dummies(test['feature_2'],prefix='feature_2',drop_first=True)
test = pd.concat([ test, frames ],axis=1)
test = test.drop(['feature_1','feature_2'],axis=1)

# Store relevant test X variables
X_test = test[['feature_3','feature_1_2','feature_1_3','feature_1_4','feature_1_5','feature_2_2','feature_2_3']]
# Apply model to test data
test_predict = rf.predict(X_test)
test_predict = pd.DataFrame(test_predict, columns=['target'])
test_predict = pd.concat([test_predict , test],axis=1)

# Stores results of model applied to test 
test_predict = test_predict[['card_id','target']]
# Export test results for scoring
test_predict.to_csv('results',index=False)
## LightGBM
import lightgbm
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor()

lgbm.fit(X, y)

y_pred = lgbm.predict(X)

mse = sklearn.metrics.mean_squared_error(y,y_pred)
rmse = np.sqrt(mse)
rmse
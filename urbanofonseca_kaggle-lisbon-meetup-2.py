# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import pearsonr

from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Visualization
import matplotlib.pyplot as plt
import seaborn as sbn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data = train_data.set_index('Id')
test_data = test_data.set_index('Id')
X, y = train_data.drop('SalePrice', axis=1), train_data['SalePrice']
X, test_X, y, test_y = train_test_split(X, y, test_size=0.2)
X.shape, test_data.shape
y.hist()
plt.title('Distribution of Sale Prices')
y.apply(lambda x: np.log(x+1)).hist()
plt.title('Log of Y')
y, test_y = np.log(y), np.log(test_y)
X.select_dtypes(include=np.number).isnull().sum().sort_values(ascending=False).head(5)
X = X.select_dtypes(include=[np.number])
test_data = test_data.select_dtypes(include=[np.number])
test_X = test_X.select_dtypes(include=[np.number])

test_data = test_data.fillna(train_data.median())
test_X = test_X.fillna(train_data.median())
X = X.fillna(X.median())
y_corr_feats = train_data.corr()['SalePrice'].abs().sort_values(ascending=False).head(20)
y_corr_feats.plot.barh()
y_corr_feats = y_corr_feats.index.tolist()
y_corr_feats
clf = RandomForestRegressor(n_estimators=100)

clf.fit(X, y)
plt.figure(figsize=(12,8))
forest_feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).plot.barh()
forest_feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
# Plot the cumulative sum
forest_feat_imp.cumsum().plot()
plt.title('Cumulative Sum of Feature Importances')
print('The total percentage of feature importances captured: %0.2f%%' % (forest_feat_imp.sum()*100))
forest_feat_imp = forest_feat_imp.index.tolist()
relevant_features = list(set(forest_feat_imp) | set(y_corr_feats))
lm = Lasso()

lm.fit(X, y)
mean_squared_error(np.log(lm.predict(test_X)), np.log(test_y))
#predictions = pd.DataFrame(np.exp(lm.predict(test_data)), columns=['SalePrice'])
#predictions['Id'] = test_data.index
predictions = pd.DataFrame(test_data.index, columns=['Id'])
predictions['SalePrice'] = np.exp(lm.predict(test_data))
predictions.to_csv('output.csv', index=False)
predictions
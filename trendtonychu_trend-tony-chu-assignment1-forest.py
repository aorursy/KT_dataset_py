import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv').fillna(0) # adding this fillna(0) because we knew beforehand there are some NA values, which are causing problems
train.describe()
# Find missing values
train.isnull().sum()[train.isnull().sum()>0]
# found Alley, FireplaceQu, PoolQC, Fence, and MiscFeature have too many missing values, they should be removed
train = train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
test = test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
# Get correlation, focusing on those > 0.5
corr = train.corr()
rel_cols = list(corr.SalePrice[(corr.SalePrice > 0.5)].index.values)
# Create matrix with independent variables
x = train[rel_cols[:-1]].iloc[:,0:].values
y = train.iloc[:, -1].values
# Create training and test dataset, split from the original train set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
# Fit Random Forest on training set
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
forest = RandomForestRegressor(n_estimators=300, random_state=0)
forest.fit(x_train, y_train)
# Score the model
forest.score(x_train, y_train)
# Predict with test set
y_pred = forest.predict(x_test)
# See how well it works
plt.figure(figsize=(12,8))
plt.plot(y_test, color='red')
plt.plot(y_pred, color='blue')
plt.show()
# Looks good... Do prediction on test data
prediction = forest.predict(test[rel_cols[:-1]].iloc[:, 0:].values)

output = pd.concat([test['Id'], pd.DataFrame(prediction)], axis=1)
output.columns = ['Id', 'SalesPrice_Prediction']
output.describe()

output.to_csv('assignment1_forest.csv', index=False)
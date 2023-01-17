# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_dataset = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train_dataset = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train_dataset
train_dataset.describe()
correlations=train_dataset.corr()

with pd.option_context('display.max_rows', None, 'display.max_columns', None): 

  print(correlations["SalePrice"].sort_values(ascending=False))
object_list = list(train_dataset.select_dtypes(include=['object']).columns)

object_list
dummies = pd.get_dummies(train_dataset[object_list])

dummies.sample(1)
train_dataset = pd.concat([train_dataset, dummies], axis=1)

train_dataset = train_dataset.drop(object_list, axis=1)

train_dataset.sample(1)
cols_with_nans=train_dataset.isna().sum()



with pd.option_context('display.max_rows', None, 'display.max_columns', None): 

  print(cols_with_nans)
train_dataset['LotFrontage'].fillna((train_dataset['LotFrontage'].mean()), inplace=True)

train_dataset['GarageYrBlt'].fillna((train_dataset['GarageYrBlt'].mean()), inplace=True)

train_dataset['MasVnrArea'].fillna((train_dataset['MasVnrArea'].mean()), inplace=True)

train_dataset.sample(1)
cols_with_nans=train_dataset.isna().sum()



with pd.option_context('display.max_rows', None, 'display.max_columns', None): 

  print(cols_with_nans)
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(train_dataset, train_size = 0.7, test_size = 0.3, random_state = 100)
df_train
X = train_dataset.iloc[:, :-2].values

X
y = train_dataset.iloc[:, train_dataset.columns == 'SalePrice']

y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression



lm = LinearRegression()

lm.fit(X_train, y_train)

y_preds = lm.predict(X)

y_preds = np.delete(y_preds, (-1), axis=0)

y_preds = [p[0] for p in y_preds]



submission_linreg = pd.DataFrame({

    "Id": test_dataset["Id"],

    "SalePrice": y_preds

})



submission_linreg



submission_linreg.to_csv('./submission_linreg.csv', index=False)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

y_pred
submission_rand_forest = pd.DataFrame({

    "Id": test_dataset["Id"],

    "SalePrice": y_pred

})

y_pred = np.delete(y_preds, (-1), axis=0)

y_pred = [p for p in y_pred]

submission_rand_forest.to_csv('./submission_linreg.csv', index=False)
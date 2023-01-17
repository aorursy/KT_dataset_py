# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.impute import SimpleImputer
fetch_from = '../input/train.csv'

train = pd.read_csv(fetch_from)
fetch_from = '../input/test.csv'

test = pd.read_csv(fetch_from)
train.head()
test.head()
train.describe()
test.describe()
train.sample(5)
train.hist(bins=50, figsize=(20,15))

plt.tight_layout(pad=0.4)

plt.show()
price = "salesPrice"
corr_matrix = train.corr()

plt.subplots(figsize=(15,10))

sns.heatmap(corr_matrix, vmax=1.0, square=True, cmap="Blues")
corr_matrix = train.corr()

corr_matrix["SalePrice"].sort_values(ascending=False)[:20]
train.isnull().sum().sum()
test.isnull().sum().sum()
test.isnull().sum()
train_fe = train.copy()

test_fe = test.copy()
train_ID = train_fe['Id']

test_ID = test_fe['Id']



train_fe.drop(['Id'], axis=1, inplace=True)

test_fe.drop(['Id'], axis=1, inplace=True)
train_fe.head()
test_fe.head()
y = train_fe["SalePrice"]
X = train_fe[["OverallQual", 'GrLivArea','GarageCars','GarageArea',"TotalBsmtSF","1stFlrSF",

              "FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]

X_pred = test_fe[["OverallQual", 'GrLivArea','GarageCars','GarageArea',"TotalBsmtSF","1stFlrSF",

              "FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]
X.head()
X_pred.head()
X_pred.isnull().sum()
X_pred.isnull().index
X_pred_nomissing = X_pred.select_dtypes(include=[np.number]).interpolate().dropna()
X_pred_nomissing.isnull().sum()
X_pred_nomissing.shape
X_pred_nomissing.head()
lm = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lm.fit(X_train,y_train)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
print(lm.intercept_)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
X_pred_results = lm.predict(X_pred_nomissing)
plt.hist(X_pred_results)
test.Id.shape
output=pd.DataFrame({'Id':test.Id, 'SalePrice':X_pred_results})

output.to_csv("submissions.csv", index=False)
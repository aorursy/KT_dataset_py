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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.SalePrice.describe()
target = df_train['SalePrice']
target_log = np.log1p(df_train['SalePrice'])
target_log.head(5)
# drop target variable from train dataset
train = df_train.drop(["SalePrice"], axis=1)
train.columns
data = pd.concat([train, df_test], ignore_index=True)
data.shape
# save all categorical columns in list
categorical_columns = [col for col in data.columns.values if data[col].dtype == 'object']
# dataframe with categorical features
data_cat = data[categorical_columns]
# dataframe with numerical features
data_num = data.drop(categorical_columns, axis=1)
data_num.head(2)
data_num.describe()
data_cat.head(2)
data_num.dropna
from scipy.stats import skew
data_num_skew = data_num.apply(lambda x: skew(x.dropna()))
data_num_skew = data_num_skew[data_num_skew > .75]

# apply log + 1 transformation for all numeric features with skewnes over .75
data_num[data_num_skew.index] = np.log1p(data_num[data_num_skew.index])
data_num_skew
data_len = data_num.shape[0]

# check what is percentage of missing values in categorical dataframe
for col in data_num.columns.values:
    missing_values = data_num[col].isnull().sum()
    #print("{} - missing values: {} ({:0.2f}%)".format(col, missing_values, missing_values/data_len*100)) 
    
    # drop column if there is more than 50 missing values
    if missing_values > 50:
        #print("droping column: {}".format(col))
        data_num = data_num.drop(col, axis = 1)
    # if there is less than 50 missing values than fill in with median valu of column
    else:
        #print("filling missing values with median in column: {}".format(col))
        data_num = data_num.fillna(data_num[col].median())
data_len = data_cat.shape[0]

# check what is percentage of missing values in categorical dataframe
for col in data_cat.columns.values:
    missing_values = data_cat[col].isnull().sum()
    #print("{} - missing values: {} ({:0.2f}%)".format(col, missing_values, missing_values/data_len*100)) 
    
    # drop column if there is more than 50 missing values
    if missing_values > 50:
        print("droping column: {}".format(col))
        data_cat.drop(col, axis = 1)
    # if there is less than 50 missing values than fill in with median valu of column
    else:
        #print("filling missing values with XXX: {}".format(col))
        #data_cat = data_cat.fillna('XXX')
        pass
data_cat.describe()
data_num.describe()
data_cat_dummies = pd.get_dummies(data_cat)

data_num.shape
train = pd.concat([df_num,df_cat] , axis = 1)
data_cat.shape
# data = pd.concat([data_num, data_cat], axis=1)
data = pd.concat([data_num, data_cat_dummies], axis=1)

train = data.iloc[:len(train)-1]
train = train.join(target_log)

test = data.iloc[len(train)+1:]
train.head(3)
# remove Id and target variable
X_train = train[train.columns.values[1:-1]]
y_train = train[train.columns.values[-1]]

# remove Id
X_test = test[test.columns.values[1:]]
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train)
#Fit a model 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500, n_jobs=-1)


model.fit(X_train1, y_train1)
predictions = model.predict(X_test1)
predictions
print("Score: ", model.score(X_test1, y_test1))
plt.figure(figsize=(10, 5))
plt.scatter(y_test1, predictions, s=20)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.plot([min(y_test1), max(y_test1)], [min(y_test1), max(y_test1)])
plt.tight_layout()
plt.show()




model.fit(X_train, y_train)
pred_log = model.predict(X_test)
pred_log
submission = pd.DataFrame({'Id':test['Id'], 'SalePrice':np.expm1(pred_log)})
submission.tail(5)
submission.to_csv("submission.csv", index=False)

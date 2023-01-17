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

from matplotlib import pyplot as plt
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
from sklearn.preprocessing import LabelEncoder



for i in range(train.shape[1]):

    if train.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))

        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))

        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))
pd.set_option('display.max_rows', 100)

print(train.isnull().any())
train = train.dropna(how='any', axis=1)

#test = test.dropna(how='any', axis=1)
test_ID = test['Id']
y_train = train['SalePrice']

x_train = train.drop(['Id','SalePrice'], axis=1)
x_test = test.drop('Id', axis=1)
log_y_train = np.log(y_train)
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=100, max_features='auto')

rf.fit(x_train, log_y_train)
sort = np.argsort(-rf.feature_importances_)

f, ax = plt.subplots(figsize=(20, 10))

sns.barplot(x=x_train.columns.values[sort],y=rf.feature_importances_[sort])

ax.set_xlabel("Feature")

ax.set_ylabel("Importance")

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
x_train = x_train.iloc[:,sort[:12]]
x_test = test[[str(x_train.columns[0]),str(x_train.columns[1]),str(x_train.columns[2]),str(x_train.columns[3]),str(x_train.columns[4]),str(x_train.columns[5]),str(x_train.columns[6]),str(x_train.columns[7]),str(x_train.columns[8]),str(x_train.columns[9]),str(x_train.columns[10]),str(x_train.columns[11])]]
print(x_test)
import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold, GridSearchCV



mod = xgb.XGBRegressor()



cv = GridSearchCV(mod,{'eta': [0.01, 0.1, 0.3, 0.5, 1.0],

                       'gamma': [0, 0.01, 0.1],

                       'max_depth': [2, 4, 6, 8],

                       'n_estimators': [10, 50, 100, 150, 200]})



cv.fit(x_train, log_y_train)



#y_train_pred = cv.predict(x_train)

y_test_pred = np.exp(cv.predict(x_test))
submission = pd.DataFrame({

    "Id": test_ID,

    "SalePrice": y_test_pred

})



submission.to_csv('houseprice.csv', index=False)
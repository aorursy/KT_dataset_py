!pip install category_encoders xgboost
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import category_encoders as ce
from xgboost import XGBRegressor

import joblib
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
submit = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
X_train, X_test, y_train, y_test = train_test_split(data.drop('SalePrice', axis=1), data.SalePrice.values, test_size=.2, random_state=233)
def feature_list(df, exclude, value_threshold):
    cat = []
    quant = []

    for i in df.columns:
        if i not in exclude:
            if df[i].dtype == 'O' or df[i].nunique() <= value_threshold:
                cat.append(i)
            else:
                quant.append(i)

    return cat, quant
cat, quant = feature_list(data, exclude=('SalePrice','Id'), value_threshold=16)
#[preprocessing training set]

cbe = ce.CatBoostEncoder(random_state=1)
X_train_cat_encoded = cbe.fit_transform(X_train[cat],y_train)
X_train_encoded = pd.concat([X_train[quant],X_train_cat_encoded],axis=1)
ss = StandardScaler()
X_train_normed = ss.fit_transform(X_train_encoded)
joblib.dump(cbe,'../model/cbe.joblib')
joblib.dump(ss,'../model/ss.joblib')
def preprocessing(X):
    ''''''
    X_cat_encoded = cbe.transform(X[cat])
    X_encoded = pd.concat([X[quant],X_cat_encoded],axis=1)
    
    X_normed = ss.transform(X_encoded)
    
    return X_normed
#[fitting model]

xgb_rgr = XGBRegressor(n_estimators=100000,max_depth=7,random_state=555)
xgb_rgr.fit(X_train_normed, y_train,eval_set=[(preprocessing(X_test),y_test)],eval_metric='rmsle',early_stopping_rounds=10)
joblib.dump(xgb_rgr,'../model/xgb_rgr.joblib')
score = pd.DataFrame({'Id': submit.Id.values, 
                      'SalePrice': xgb_rgr.predict(preprocessing(submit))
                     })
score.head()
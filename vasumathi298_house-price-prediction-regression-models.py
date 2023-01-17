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
df_train  = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')   
df_test.isnull().sum().sort_values(ascending = False)[0:40]
for col in df_train:
    if df_train[col].dtype == 'object':
      df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
    else:
       df_train[col].fillna(round(df_train[col].mean()),inplace = True)


for col in df_test:
      if df_test[col].dtype == 'object':
          df_test[col] = df_test[col].fillna(df_test[col].mode()[0])
      else:
        df_test[col].fillna(round(df_test[col].mean()),inplace = True)
df_train.drop(columns = ['Id', ],axis = 1, inplace =True)
df_test.drop(columns = ['Id'],axis = 1, inplace =True)
def cat_onehotencoder(df_concat):
    df_temp = df_concat
    for col in df_temp:
        if df_temp[col].dtype =='object':
            df1 = pd.get_dummies(df_concat[col], drop_first = True)
            df_concat.drop([col], axis = 1, inplace = True)
            
            df_concat = pd.concat([df_concat,df1], axis = 1)
        
    
        
    
    return df_concat
y = df_train.iloc[:,-1].values
df_t = df_train
y
df_train.drop(columns = ['SalePrice'], axis = 0, inplace = True)
df_concat = pd.concat([df_train,df_test], axis = 0)
df_final =  cat_onehotencoder(df_concat)
df_final =df_final.loc[:,~df_final.columns.duplicated()]
df_final.shape
import seaborn as sns
correlations = df_train[df_train.columns].corr(method='pearson')
sns.heatmap(correlations, annot = True)


train = df_final.iloc[:1460,:]
test = df_final.iloc[1460:,:]
X= train.iloc[:,:].values
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
from xgboost import XGBRegressor
reg_xgb = XGBRegressor()
reg_xgb.fit(X_train,y_train)
ypred_xgb = reg_xgb.predict(X_val)

score_xgb = r2_score(ypred_xgb, y_val)
MSL_xgb = mean_squared_log_error(ypred_xgb,y_val)
print(score_xgb, MSL_xgb)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_log_error
GB_reg = GradientBoostingRegressor()
GB_reg.fit(X_train, y_train)

y_predgb = GB_reg.predict(X_val)
score_gb = r2_score(y_predgb, y_val)
MSL_gb = mean_squared_log_error(y_predgb,y_val)
print(score_gb, MSL_gb)
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators =75, random_state = 42)
regressor_rf.fit(X_train,y_train)
ypred_rf = regressor_rf.predict(X_val)
score_rf = r2_score(ypred_rf, y_val)
MSL_rf = mean_squared_log_error(ypred_rf,y_val)
print(score_rf, MSL_rf)
from sklearn import tree
regressor_rf = tree.DecisionTreeRegressor()
regressor_rf.fit(X_train,y_train)
ypred_rf = regressor_rf.predict(X_val)
score_rf = r2_score(ypred_rf, y_val)
MSL_rf = mean_squared_log_error(ypred_rf,y_val)
print(score_rf, MSL_rf)
y_pred_final = GB_reg.predict(sc.fit_transform(test))
sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
SalePrice = pd.DataFrame(y_pred_final, columns = ['SalePrice'])
len(SalePrice)
SalePrice.insert(0, 'Id', sub['Id'], True)
SalePrice.to_csv('sample_submissions1.csv', index = False)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import xgboost
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    LinearRegression,
    ElasticNet,
    Ridge,
    Lasso
)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#učitavanje podataka
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#PREDPROCESIRANJE (izdvojit u poseban razred)

#korelacija značajki
corr = train.corr()
plt.subplots(figsize=(12,12))
sns.heatmap(corr)
#korelacija značajki koje su najviše korelirane s cijenom 
k = 10 
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
hm = sns.heatmap(cm, annot=True, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#GarargeArea i GarageCars su visoko korelirani pa možemo izbrisat jednu zbog multikolinearnosti
#Isto je i za 1stFlrSF i TotalBsmtSF
test.drop(['GarageArea'], axis = 1,inplace=True)
train.drop(['GarageArea'], axis = 1,inplace=True)
test.drop(['1stFlrSF'], axis = 1,inplace=True)
train.drop(['1stFlrSF'], axis = 1,inplace=True)
#id je nebitan za treniranje i predikciju
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#provjera koliko podataka nedostaje
train, y_train = train[train.columns[:-1]], train[train.columns[-1]]
alldata = pd.concat([train,test],axis=0).reset_index(drop=True)
alldata.isnull().sum()[alldata.isnull().sum()>0].sort_values(ascending=False)
#uklanjanje stupaca s najviše vrijednosti koje nedostaju
alldata.drop(['PoolQC','MiscFeature','Fence','Alley'], axis = 1,inplace=True)
test.drop(['PoolQC','MiscFeature','Fence','Alley'], axis = 1,inplace=True)
train.drop(['PoolQC','MiscFeature','Fence','Alley'], axis = 1,inplace=True)

#postavljanje vrijednosti koje nedostaju na srednju vrijednost stupca
#alldata = alldata.fillna(alldata.mode().iloc[0])
#test = test.fillna(test.mode().iloc[0])
#train = train.fillna(train.mode().iloc[0])

#bolje rezultate daje postavljanje na 0/None (za većinu značajki)
#za functional piše da NA znači Typical
alldata["Functional"] = alldata["Functional"].fillna("Typ")
alldata['MSZoning'] = alldata['MSZoning'].fillna(alldata['MSZoning'].mode()[0])

nan = alldata.isnull().sum()[alldata.isnull().sum()>0].index.tolist() 
nan_float = alldata[nan].dtypes[alldata[nan].dtypes=='float64'].index.tolist() 
nan_obj = alldata[nan].dtypes[alldata[nan].dtypes=='object'].index.tolist() 
for nan_float in nan_float:
    alldata.loc[alldata[nan_float].isnull(),nan_float] = 0.0
for nan_obj in nan_obj:
    alldata.loc[alldata[nan_obj].isnull(),nan_obj] = 'None'
#kodiranje kategorickih značajki koje su međusobno usporedive
cols = ('ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 
        'BsmtFinType2', 'Functional','BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive',  'CentralAir', 'MSSubClass', 'OverallCond','YrSold',
        'MoSold','LotShape','FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond','BsmtFinType1',
      )
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(alldata[c].values))  
    alldata[c] = lbl.transform(list(alldata[c].values))
#kodiranje značajki koje nisu međusobno usporedive
cols_one_hot = ('MSZoning','Street','LandContour','Utilities','LotConfig','Neighborhood','Condition1','Condition2',
               'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
               'Foundation','Heating','Electrical','GarageType','SaleType','SaleCondition')
transformer1 = ColumnTransformer(
    transformers=[
        ("Houses",        
         OneHotEncoder(), 
         cols_one_hot  
         )
    ], remainder='passthrough'
)
transformer1.fit(alldata)
alldata=transformer1.transform(alldata)

train = alldata[:train.shape[0]]
test = alldata[train.shape[0]:]
print(train.shape)
#isprobala sam pca jer ima puno značajki, ali ne daje bolje rezultate..
scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)
test=scaler.transform(test)
#pca = PCA(n_components = 0.95)
#pca.fit(train)
#train = pca.transform(train)
#test=pca.transform(test)
sns.distplot(y_train, fit=norm);
fig = plt.figure()
#izlaz se nije ravnao po normalnoj razdiobi pa ga možemo logaritmirat 
#model će davat bolje rezultate za normalnu razdiobu
y_train = np.log(y_train)
sns.distplot(y_train, fit=norm);
fig = plt.figure()
#MODELI
#prvo sam probala osnovne linearne modele s regularizacijom jer uvijek prvo biram najjednostavnije
#oni nisu dovolji za ovaj problem, a ni njihova kombinacija nije dala baš dobre rezultate..
#nakon toga probala XGBReggresor i LGBMRegressor jer znam da XGBReggresor često daje najbolje 
#rezultate ta ovakve probleme
#možda bi za bolje rezultate još isprobala neuronsku mrežu ali mislim da je zbog brzine XGBReggresor 
#dovoljno dobar izbor :)

#isprobavanje baseline modela - linearni + l2 regularizacija 
model = Ridge(alpha=1)
model.fit(train,y_train)
y_pred=model.predict(train)
print(np.sqrt(mean_squared_error(y_train, y_pred)))
#lasso - linearni + l1 regularizacija, značajke su korelirane pa ima smisla l1
model = Lasso(alpha=0.00005,max_iter=10000)
model.fit(train,y_train)
y_pred=model.predict(train)
print(np.sqrt(mean_squared_error(y_train, y_pred)))
#elasticnet - linearni +l1+l2
model = ElasticNet(alpha=0.005,l1_ratio=0.0005,max_iter=10000)
model.fit(train,y_train)
y_pred=model.predict(train)
print(np.sqrt(mean_squared_error(y_train, y_pred)))
#xgbregressor
model = xgboost.XGBRegressor(learning_rate=0.05,n_estimators=2048,
                                      min_child_weight=1,
                                      subsample=0.5,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.06,reg_lambda=1)
#X_train, X_test, y_t, y_test = train_test_split(train, y_train, test_size=0.33, random_state=42)
model.fit(train, y_train)
predictions = np.exp(model.predict(test))
y_pred=model.predict(train)
print(np.sqrt(mean_squared_error(y_train, y_pred)))
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission['SalePrice'] = predictions
submission.to_csv('outputxgb.csv',index=False)
mod1=Lasso(alpha=0.0005,max_iter=10000)
mod2=Ridge(alpha=1)
mod3=ElasticNet(alpha=0.05,l1_ratio=0.005,max_iter=10000)
mod4 = xgboost.XGBRegressor(learning_rate=0.05,n_estimators=2048,
                                      min_child_weight=1,
                                      subsample=0.5,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.06,reg_lambda=1)
ereg = VotingRegressor(estimators=[('lasso', mod1), ('ridge', mod2), ('en', mod3),('xgb',mod4)])
ereg = ereg.fit(train, y_train)
y_pred=ereg.predict(train)
print(np.sqrt(mean_squared_error(y_train, y_pred)))
model = RandomForestRegressor(n_estimators=200)
model.fit(train,y_train)
y_pred=model.predict(train)
print(np.sqrt(mean_squared_error(y_train, y_pred)))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=11,
                              learning_rate=0.055, n_estimators=2024,
                              max_bin = 55, bagging_fraction = 0.9,
                              bagging_freq = 5, feature_fraction = 0.3,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =5, min_sum_hessian_in_leaf = 13)
X_train, X_test, y_t, y_test = train_test_split(train, y_train, test_size=0.33, random_state=42)
model_lgb.fit(X_train, y_t)
predictions = model_lgb.predict(test)
y_pred=model_lgb.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))
model = xgboost.XGBRegressor(learning_rate=0.05,n_estimators=2048,
                                      min_child_weight=1,
                                      subsample=0.5,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.06,reg_lambda=1)
model.fit(train, y_train)
predictionsxgb = np.exp(model.predict(test))

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=11,
                              learning_rate=0.055, n_estimators=2024,
                              max_bin = 55, bagging_fraction = 0.9,
                              bagging_freq = 5, feature_fraction = 0.3,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =5, min_sum_hessian_in_leaf = 13)
model_lgb.fit(train, y_train)
predictionslgb = np.exp(model_lgb.predict(test))
predictions=0.6*predictionsxgb+0.4*predictionslgb
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission['SalePrice'] = predictions
submission.to_csv('outputkomb.csv',index=False)
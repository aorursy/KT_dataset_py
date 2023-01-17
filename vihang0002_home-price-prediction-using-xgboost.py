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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import tree

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df1=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
df.shape
df1.shape
df2=pd.concat([df,df1], ignore_index=True, axis=0)
df2.shape
train_index=df.shape[0]
sns.heatmap(df2.isnull(),yticklabels=False)
df2.dropna(thresh=df2.shape[0]*0.3,how='all',axis=1,inplace=True)#drops columns who have atleast 30% of their inputs as NaN 
pd.set_option('display.max_columns', None)
df2.drop(["Id"],axis=1,inplace=True)
for features in df2.columns:
    if df[features].isnull().sum() >=1:
        print(features, df2[features].isnull().sum(),"missing values" )
df2.drop(["FireplaceQu"],axis=1,inplace=True)
df2["LotFrontage"].fillna(df["LotFrontage"].median(),inplace=True)
df2['BsmtCond'] = df2['BsmtCond'].fillna(df2.BsmtCond.mode()[0])
df2['BsmtQual'] = df2['BsmtQual'].fillna(df2.BsmtCond.mode()[0])
null_features=[]
for features in df2.columns:
    if df[features].isnull().sum() >=1:
        null_features.append(features)
for item in null_features:
    df2[item] = df2[item].fillna(df2[item].mode()[0])
sns.heatmap(df2.isnull(),yticklabels=False)
numerical_features = [feature for feature in df2.columns if df2[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

discrete_feature=[feature for feature in numerical_features if len(df2[feature].unique())<25 ]
print("Discrete Variables Count: {}".format(len(discrete_feature)))
for feature in discrete_feature:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature]
print("Continuous feature Count {}".format(len(continuous_feature)))
for feature in continuous_feature:
    data=df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()
for feature in continuous_feature:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()
categorical_features=[feature for feature in df2.columns if df2[feature].dtypes=='O']
for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(df2[feature].unique())))
print("Categorical Features Count: {}".format(len(categorical_features)))
numerical_with_nan=[feature for feature in numerical_features if df2[feature].isnull().sum()>1]
for feature in numerical_with_nan:
    print("{}: {}% missing value".format(feature,np.around(df2[feature].isnull().mean(),4)))
df2["BsmtFullBath"].fillna(df2["BsmtFullBath"].mode())
df2["BsmtHalfBath"].fillna(df2["BsmtHalfBath"].mode())
for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       
    df2[feature]=df2['YrSold']-df2[feature]
features_with_nan=[]
for features in df2.columns:
    if df2[features].isnull().sum() >=1:
        features_with_nan.append(features)
        print(features, df2[features].isnull().sum(),"missing values" )
for features in features_with_nan:
    if features!="SalePrice":
        df2[features]=df2[features].fillna(df2[features].mode()[0])
df2.shape
for feature in categorical_features:
    T=df2[feature].value_counts()
    less_than_10=T[T<=10]
    df2[feature]=df2[feature].apply(lambda x:"other" if x in less_than_10 else x)
        
df_corr=df2[numerical_features].corr()
df_corr["SalePrice"].sort_values(ascending=False)[:10]
df_corr["SalePrice"].sort_values(ascending=False)[-10:]
corr_df = pd.DataFrame(df_corr['SalePrice'].sort_values(ascending=False))
corr_df = corr_df.reset_index()
corr_df.columns = ['cols', 'values']
cols_to_drop = list(corr_df[corr_df['values'] <0]['cols'])
df2.drop(labels=cols_to_drop, axis=1,inplace=True)
df2.shape
for features in df2.columns:
    if df2[features].isnull().sum() >=1:
        features_with_nan.append(features)
        print(features, df2[features].isnull().sum(),"missing values" )
y=df2["SalePrice"]
X=df2.drop("SalePrice",axis=1)
print(X.shape)
print(y.shape)
X1=pd.get_dummies(X,drop_first=True)
X1.head()
continuous_featureX=[feature for feature in numerical_features if feature in X.columns]
for feature in continuous_featureX:   
    if 0 in X[feature].unique():
        pass
    else:
        X1[feature]=np.log(X1[feature])
            
X1.head()
X1_train=X1[:train_index]
X1_test=X1[train_index:]
y_train=y[:train_index]
y_test=y[train_index:]

import xgboost as xgb
from xgboost import XGBClassifier
print(X1_train.shape)
print(X1_test.shape)
print(y_train.shape)
print(y_test.shape)
X11_train,X11_test,y11_train,y11_test=train_test_split(X1_train,y_train,test_size=0.15,random_state=42)
xgb = XGBRegressor(booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0,
             importance_type='gain', max_delta_step=0,
             n_jobs=1, nthread=None, objective='reg:squarederror',
             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 
             silent=None, subsample=0.8, verbosity=1, learning_rate=0.1,
             n_estimators=1000)

xgb.fit(X11_train, y11_train)
#preds=xgb.predict(X1_test)
xgb.score(X11_test,y11_test)
xgb.fit(X1_train, y_train)
preds=xgb.predict(X1_test)
sub_csv = pd.DataFrame({
        "Id": df1["Id"],
        "SalePrice": preds
    })
sub_csv.to_csv('submission1.csv', index=False)

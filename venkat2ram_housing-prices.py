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
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df.head()
df_test.head()
sb.heatmap(df.isnull(),yticklabels=False,cbar=False)
sb.heatmap(df_test.isnull(),cbar=False)
df_test.shape

df.shape
df.info()
df_test.info()
df_test['MSZoning']=df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df_test['LotFrontage']=df_test['LotFrontage'].fillna(df['LotFrontage'].mean()) 
df.drop('Alley', axis=1, inplace=True)
df_test.drop('Alley',axis=1,inplace=True)
df_test['Utilities']=df_test['Utilities'].fillna(df['Utilities'].mode()[0]) 
df_test['Exterior1st']=df_test['Exterior1st'].fillna(df['Exterior1st'].mode()[0]) 
df_test['Exterior2nd']=df_test['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0]) 
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df_test['MasVnrType']=df_test['MasVnrType'].fillna(df['MasVnrType'].mode()[0])  
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
df_test['MasVnrArea']=df_test['MasVnrArea'].fillna(df['MasVnrArea'].mean())   
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df_test['BsmtQual']=df_test['BsmtQual'].fillna(df['BsmtQual'].mode()[0])  
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0]) 
df_test['BsmtCond']=df_test['BsmtCond'].fillna(df['BsmtCond'].mode()[0])  
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])  
df_test['BsmtExposure']=df_test['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])  
df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])  
df_test['BsmtFinType1']=df_test['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])  
df_test['BsmtFinSF1']=df_test['BsmtFinSF1'].fillna(df['BsmtFinSF1'].mean())  
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])   
df_test['BsmtFinType2']=df_test['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])  
df_test['BsmtFinSF2']=df_test['BsmtFinSF2'].fillna(df['BsmtFinSF2'].mean())  
df_test['BsmtUnfSF']=df_test['BsmtUnfSF'].fillna(df['BsmtUnfSF'].mean())  
df_test['TotalBsmtSF']=df_test['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mean())  
df_test['BsmtFullBath']=df_test['BsmtFullBath'].fillna(df['BsmtFullBath'].mean())  
df_test['BsmtHalfBath']=df_test['BsmtHalfBath'].fillna(df['BsmtHalfBath'].mean())  
df_test['KitchenQual']=df_test['KitchenQual'].fillna(df['KitchenQual'].mode()[0])  
df_test['Functional']=df_test['Functional'].fillna(df['Functional'].mode()[0])  
df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])    
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])    
df_test['FireplaceQu']=df_test['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])  
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])    
df_test['GarageType']=df_test['GarageType'].fillna(df['GarageType'].mode()[0])    
df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0])    
df_test['GarageYrBlt']=df_test['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())  
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])    
df_test['GarageFinish']=df_test['GarageFinish'].fillna(df['GarageFinish'].mode()[0])    
df_test['GarageCars']=df_test['GarageCars'].fillna(df['GarageCars'].mean())  
df_test['GarageArea']=df_test['GarageArea'].fillna(df['GarageArea'].mean())  
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])    
df_test['GarageQual']=df_test['GarageQual'].fillna(df['GarageQual'].mode()[0])    
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])    
df_test['GarageCond']=df_test['GarageCond'].fillna(df['GarageCond'].mode()[0])   
df.drop('PoolQC',axis=1,inplace=True)
df_test.drop('PoolQC',axis=1,inplace=True)
df.drop('Fence',axis=1,inplace=True) 
df_test.drop('Fence',axis=1,inplace=True) 
df.drop('MiscFeature',axis=1,inplace=True) 
df_test.drop('MiscFeature',axis=1,inplace=True) 
df_test['SaleType']=df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])   
#find categorical features.
def find_cate(df):
    cols=df.columns
    i=0
    cat_cols=[]
    for col in cols:
        if len(df[col].unique())<=25 and df[col].dtypes!='int64' :
            cat_cols.append(col)
            i=i+1
    print(i)
    return cat_cols
            
cat_cols=find_cate(df)
#get dummies for categorical features.
def get_multcols(df,cat_cols):
    for col in cat_cols:
        df1=pd.get_dummies(df[col],drop_first=True)
        df=pd.concat([df,df1],axis=1)
        df.drop(col,axis=1,inplace=True)
    return df
main_df=get_multcols(df,cat_cols)
total_df=pd.concat([df_test,df],sort=False)
total_df1=get_multcols(total_df,cat_cols)
main_df=main_df.loc[ :,~main_df.columns.duplicated()]
total_df1=total_df1.loc[:,~total_df1.columns.duplicated()]
main_df.shape
X=main_df.drop('SalePrice',axis=1)
y=main_df['SalePrice']
X_test=total_df1.drop('SalePrice',axis=1)[0:1459]
import xgboost
classifier=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
classifier.fit(X,y)
y_pred=classifier.predict(X_test)
y_pred
y_sub=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
y_sub=pd.concat([y_sub['Id'],pd.DataFrame(y_pred)],axis=1)
y_sub.columns=['Id','SalePrice']
y_sub.to_csv('submission.csv',index=False)
len(y_pred)
len(y_pred1)
#////////////////////////////////////////////////////////////////////
#(classifier.predict(X)-y)*(classifier.predict(X)-y)
sum((classifier.predict(X)-y)*(classifier.predict(X)-y))
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(X,y)
y_pred1=lg.predict(X_test)

y_sub=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
y_sub=pd.concat([y_sub['Id'],pd.DataFrame(y_pred1)],axis=1)
y_sub.columns=['Id','SalePrice']
y_sub.to_csv('submission.csv',index=False)
y_sub
a=main_df.corr()
a_index=pd.DataFrame(a.columns)
a_index.index=a.index
a_index.columns=['ind']
c=pd.concat([a['SalePrice'],a_index],axis=1)
d=c.set_index('ind').T.to_dict('list')
corr_col=[]
for key in d:
    if (d[key][0]>=0.5) and key!='SalePrice' :
        corr_col.append(key)
X_corr=X[corr_col]
lg1=LogisticRegression(max_iter=1000000)
lg1.fit(X_corr,y)
sum((lg1.predict(X_corr)-y)*(lg1.predict(X_corr)-y))
c.shape
pd.DataFrame(a.index)
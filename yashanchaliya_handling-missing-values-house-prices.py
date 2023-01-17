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
%matplotlib notebook
import matplotlib.pyplot as plt

df1=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
               
df1.head()
def dataframe(df):
    missing_value=df.isnull().sum()
    missing_percentage=100*(missing_value/len(df))
    new_df=pd.concat([missing_value,missing_percentage],axis=1)
    new_df_rename=new_df.rename(columns={0:'no of missing values',1:'% of missing values'})
    new_new_df=new_df_rename[new_df_rename.iloc[:,0]!=0].sort_values(by='% of missing values',ascending=False).round(1)
    return new_new_df
df2=dataframe(df1)
df2
df1.head()
df1.drop(columns=['Id','PoolQC','MiscFeature','Alley','Fence'],inplace=True)
df1
import missingno as msno
df_missing=df1[['FireplaceQu','LotFrontage',"GarageType",'GarageYrBlt','GarageFinish','GarageQual','GarageCond','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrArea','MasVnrType','Electrical']]
df_missing
msno.bar(df_missing)
msno.heatmap(df_missing)
msno.dendrogram(df_missing)
msno.matrix(df_missing)
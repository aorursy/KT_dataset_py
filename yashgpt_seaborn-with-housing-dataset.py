import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/housetrain.csv',index_col=0)
df.head()
df.tail()
df.columns
df.shape
df.size
df.info
df1=df._get_numeric_data()
df1
df1=df._get_numeric_data().columns
nfc=list(df1)
nfc
df2=df.columns
dd=list(df2)
a=set(nfc)
b=set(dd)
b
b-a
sns.distplot(df['SalePrice'])
sns.distplot(df[nfc[0]],kde=False) #univariable for numeric also boxplot 
sns.countplot('HalfBath',data=df)
sns.countplot(y='FullBath',data=df)   #univarient for categorical
sns.lmplot('GrLivArea','SalePrice',data=df,fit_reg=True) #multivarient for numeric
sns.jointplot('GrLivArea','SalePrice',data=df,kind='reg')
sns.jointplot('GrLivArea','SalePrice',data=df,kind='hex')
#categorical vs categorical
crosstab=pd.crosstab(index=df["Neighborhood"],columns=df['OverallQual'])
crosstab
crosstab.plot(kind='bar',figsize=(12,8),stacked=True,colormap='Paired')
cols=['SalePrice','OverallQual','GrLivArea','GarageCars']
sns.pairplot(df[cols])
sns.heatmap(df.corr(),cmap='viridis')
sns.boxplot('Neighborhood','SalePrice',data=df) #categorical vs numeric
sns.swarmplot('Street','SalePrice',data=df) 
sns.violinplot('Street','SalePrice',data=df)
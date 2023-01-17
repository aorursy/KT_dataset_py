#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#loading data
train_df = pd.read_csv('../input/train.csv')
train_df.describe()
# important numeric features
train_df_int = train_df.select_dtypes(exclude = 'object')
#correlation matrix
corrmat = train_df_int.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df_int[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
high_corr_cols = ['TotRmsAbvGrd','GarageCars','1stFlrSF']
cols = [i for i in cols if i not in high_corr_cols]
train_df_fin = train_df[cols]
train_df_obj = train_df.select_dtypes(include='object')
train_df_obj.head()
#box plot overallqual/saleprice
var = 'Neighborhood'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
# cols_keep = ['Neighborhood','HouseStyle','ExterQual','CentralAir', 'LotShape','HeatingQC','PoolQC']
lotshape_map = {'Reg':1, 'IR1':2,'IR2':3,'IR3':4}
col_bin = 'Neighborhood'
neighborhood_df = train_df.groupby(col_bin).agg({'SalePrice':'median',
                                     'Id':'count'}).sort_values(by ='SalePrice', ascending=False)
neighborhood_df.reset_index(inplace=True)
neighborhood_df['neighbor_bins'] = pd.cut(neighborhood_df['SalePrice'],bins=5, labels=range(1,6))
neighbor_bins = neighborhood_df[[col_bin,'neighbor_bins']].drop_duplicates()
neighbor_dict = dict(zip(neighbor_bins.Neighborhood, neighbor_bins.neighbor_bins))
col_bin = 'HouseStyle'
neighborhood_df = train_df.groupby(col_bin).agg({'SalePrice':'median',
                                     'Id':'count'}).sort_values(by ='SalePrice', ascending=False)
neighborhood_df.reset_index(inplace=True)
neighborhood_df['style_bins'] = pd.cut(neighborhood_df['SalePrice'],bins=5, labels=range(1,6))
style_bins = neighborhood_df[[col_bin,'style_bins']].drop_duplicates()
style_dict = dict(zip(style_bins.HouseStyle,style_bins.style_bins))

col_bin = 'ExterQual'
neighborhood_df = train_df.groupby(col_bin).agg({'SalePrice':'median',
                                     'Id':'count'}).sort_values(by ='SalePrice', ascending=False)
neighborhood_df.reset_index(inplace=True)
neighborhood_df['exqual_bin'] = pd.cut(neighborhood_df['SalePrice'],bins=2, labels=range(1,3))
exqual_bin = neighborhood_df[[col_bin,'exqual_bin']].drop_duplicates()
exqual_dict = dict(zip(exqual_bin.ExterQual,exqual_bin.exqual_bin))

heating_bins = {'Po':0,'Fa':1,'TA':1,'Gd':2,'Ex':3}
central_air_bins = {'Y':1,'N':0}
pool_bins = {'Gd':1,'Fa':2,'Ex':3}
neighbor_dict = dict(zip(neighbor_bins.Neighborhood, neighbor_bins.neighbor_bins))
neighbor_dict
exqual_bin.to_dict()

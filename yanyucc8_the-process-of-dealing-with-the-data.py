 #the module we use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

pd.set_option('display.float_format', lambda x:'{:.3f}'.format(x))
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# input the dataset
df_train = pd.read_csv('../input/train.csv')
# check the columns' name
df_train.columns
df_train.head()
print('The train data size before dropping id feature is :{}'.format(df_train.shape))

df_train_id = df_train['Id']
df_train.drop('Id', axis = 1, inplace = True)

print('The train data after dropping id feature is :{}'.format(df_train.shape))
#describe the saleprice
round(df_train['SalePrice'].describe())
sns.distplot(df_train['SalePrice'])
print('The skewness of the saleprice is : %f' % df_train['SalePrice'].skew())
print('The kurtosis of the saleprice is : %f' % df_train['SalePrice'].kurt())
corrmat = df_train.corr()
f,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat,vmax = .8,square = True)
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
f,ax = plt.subplots(figsize = (9,6))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();
#The most familiar way to visualize a bivariate distribution is a scatterplot, 
#where each observation is shown with point at the x and y values. 
sns.jointplot(x = df_train['GrLivArea'],y = df_train['SalePrice'],data = df_train,alpha = .6)
# drop the outliers' index
train = df_train.drop(df_train[(df_train['SalePrice'] < 200000) & (df_train['GrLivArea'] > 4000)].index)
#view the new train data：SalePrice～GrLivArea
sns.jointplot(x = train['GrLivArea'],y = train['SalePrice'],data = train, kind= 'reg')
#view the new train data: SalePrice~TotalBsmtSF
sns.jointplot(x = train['TotalBsmtSF'],y = train['SalePrice'],data = train ,kind = 'reg')
#SalePrice~OverallQual

f,ax = plt.subplots(figsize = [8,6])
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
sns.violinplot(x = 'OverallQual',y = 'SalePrice',data = train)
#SalePrice ~ GarageCars

f, ax = plt.subplots(figsize = [8,6])
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
sns.violinplot(x = 'GarageCars', y = 'SalePrice', data = train)
from scipy import stats

#group the data
t0 = train[train['GarageCars']==0]['SalePrice']
t1 = train[train['GarageCars']==1]['SalePrice']
t2 = train[train['GarageCars']==2]['SalePrice']
t3 = train[train['GarageCars']==3]['SalePrice']
t4 = train[train['GarageCars']==4]['SalePrice']

args = [t0,t1,t2,t3,t4]

#levene test
w,p = stats.levene(*args)
if p<0.05:
    print (u'Warning: Variance of Levene Test is uneven (p = %.2f)' % p)
    
f,p = stats.f_oneway(*args)
print (f,p)
# anova in pandas
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
anova_result = anova_lm(ols('SalePrice ~ C(GarageCars)',train).fit())
print(anova_result)
#SalePrice ~ FullBath

f, ax = plt.subplots(figsize = [8,6])
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
sns.violinplot(x = 'FullBath', y = 'SalePrice',data = train)
total = train.isnull().sum().sort_values(ascending = False)
total = total.drop(total[total == 0].index)
ratio = ((train.isnull().sum() / len(train)) * 100).sort_values(ascending = False)
ratio = ratio.drop(ratio[ratio == 0].index)
missing_data = pd.concat([total,ratio],axis = 1,keys = ['Total','Percent'])
missing_data.head(20)
f,ax = plt.subplots(figsize = [12,9])
plt.xticks(rotation='90')
sns.barplot(x = ratio.index ,y = ratio)
plt.xlabel('Features',fontsize = 15)
plt.ylabel('Percent of missing values',fontsize = 15)
train['PoolQC'] = train['PoolQC'].fillna('None')
train['MiscFeature'] = train['MiscFeature'].fillna('None')
train['Alley'] = train['Alley'].fillna('None')
train['Fence'] = train['Fence'].fillna('None')
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x : x.fillna(x.median()) )
for column in ['GarageType','GarageFinish','GarageQual','GarageCond']:
    train[column] = train[column].fillna('None')
for column in ['GarageYrBlt', 'GarageArea' , 'GarageCars']:
    train[column] = train[column].fillna(0)
for column in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    train[column] = train[column].fillna(0)
for column in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1' , 'BsmtFinType2']:
    train[column] = train[column].fillna('None')
train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
train['MasVnrType'] = train['MasVnrType'].fillna('None')
sns.countplot(x = 'Electrical',data = train)
#df.mode() : get the most frequent value in a column
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0]) 
#Check remaining missing values if any 
total = train.isnull().sum().sort_values(ascending = False)
total = total.drop(total[total == 0].index)
ratio = ((train.isnull().sum() / len(train)) * 100).sort_values(ascending = False)
ratio = ratio.drop(ratio[ratio == 0].index)
missing_data = pd.concat([total,ratio],axis = 1,keys = ['Total','Percent'])
missing_data.head()
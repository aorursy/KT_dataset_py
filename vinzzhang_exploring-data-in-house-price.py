import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
df_train = pd.read_csv('../input/train.csv')
df_train.shape
df_train.head()
df_train["SalePrice"].describe()  #desciptive statistics summary()
hist = sns.distplot(df_train["SalePrice"])  #Histogram
# Scatter plot with regression line
f, ax = plt.subplots(figsize=(7,5))
fig = sns.regplot(df_train["GrLivArea"], df_train["SalePrice"], scatter_kws={'s':10})
# Scatter plot with regression line
f, ax = plt.subplots(figsize=(7,5))
fig = sns.regplot(df_train["TotalBsmtSF"], df_train["SalePrice"], scatter_kws={'s':10})
# Boxplot 
dataset = pd.concat([df_train["SalePrice"], df_train["OverallQual"]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x="OverallQual", y="SalePrice", data=dataset)
# Boxplot 
dataset2 = pd.concat([df_train["YearBuilt"], df_train["SalePrice"]], axis=1)
f,ax = plt.subplots(figsize=(16,10))
fig = sns.boxplot(x="YearBuilt", y="SalePrice", data=dataset2)
corrMatrix = df_train.corr() #Compute pairwise correlation of columns, excluding NA/null values
cols = corrMatrix.nlargest(10,'SalePrice')['SalePrice'].index
cols
cm = np.corrcoef(df_train[cols].values.T) #Return Pearson product-moment correlation coefficients. .T:transpose
sns.set(font_scale=1.2)
f, ax = plt.subplots(figsize=(7,5))
hm = sns.heatmap(cm, cbar=True, annot=True, annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
pp = sns.pairplot(df_train[cols], size = 2.75)
total= df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/len(df_train)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
missing_data = missing_data[missing_data['Total'] != 0 ]
missing_data
# Delete features with MissingData percent over 10%
df_train = df_train.drop((missing_data[missing_data['Percent'] > 0.1]).index, axis=1)
df_train.shape
# Fill means in numerical missed columns; modes in categorical missed columns
def fill_missing_data(x):
    if x.dtype != "object":
        x = x.fillna(x.mean())
    else:
        x = x.fillna(x.mode()[0])
    return x

to_be_filled = missing_data[missing_data['Percent'] < 0.1].index

df_train[to_be_filled] = df_train[to_be_filled].apply(lambda x: fill_missing_data(x))
# Is null checking
df_train.isnull().any().any()
#Histogram and Normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#Histogram and Normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
dist = sns.distplot(df_train['GrLivArea'], fit=norm)
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#transformed histogram and normal probability plot
dist = sns.distplot(df_train['GrLivArea'], fit=norm)
#histogram and normal probability plot
dis = sns.distplot(df_train['TotalBsmtSF'], fit=norm)
df_train['HasBsmt'] = pd.Series(df_train['TotalBsmtSF'], index = df_train.index)
df_train['HasBsmt'] = 0
# if TotalBsmtSF > 0, HasBsmt = 1; else HasBsmt = 0 
df_train['HasBsmt'] = np.where(df_train['TotalBsmtSF'] > 0, 1, 0)
# df_train[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
df_train['TotalBsmtSF'] = np.where(df_train['HasBsmt'] == 1, np.log(df_train['TotalBsmtSF']), 0)
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm) #df_train[df_train['TotalBsmtSF']>0] is dataframe !!!

df_train = df_train[df_train['HasBsmt'] == 1]
dis = sns.distplot(df_train['YearBuilt'], fit=norm)
print(stats.levene(df_train['GrLivArea'], df_train['SalePrice']))
sc = plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
print(stats.levene(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']))
sc = plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice'])
df_train = pd.get_dummies(df_train)




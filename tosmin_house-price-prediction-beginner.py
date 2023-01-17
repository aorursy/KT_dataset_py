import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from scipy.stats import pearsonr

import statsmodels.api as sm

from sklearn.preprocessing import scale

from scipy import stats

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/home-data-for-ml-course/train.csv')
df.head(6)
df.tail(6)
df.info()
df.columns
df.shape
df.describe().style.background_gradient(cmap='Blues')
df.skew()
df.kurt()
df.isna().sum().sort_values()
#total no of missing value

df.isna().sum().sum()
mis = df.isnull().sum().to_frame()



mis.columns = ['nMissings']





mis['perMissing'] = mis['nMissings']/1460

mis = mis[mis.nMissings >= 1]



misor = mis.sort_values(by = ['nMissings'], ascending=False)

plt.figure(figsize=(30,10))          

sns.barplot(x = misor.index, y = misor['perMissing']);

plt.xticks(rotation=90);
print(df['SalePrice'].describe())
#distribution plot of sales price

fig, ax=plt.subplots(figsize=(30,10))

sns.distplot(a=df['SalePrice'], ax=ax);
sns.boxplot(data=df['SalePrice']);
sns.scatterplot(df['SalePrice'], df['YrSold']);
var = ['SalePrice', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']
plt.figure(figsize=(30,10))

corr = df[var].corr()

sns.heatmap(data=corr, annot=True);
print('correlation in betwn saleprice and yearsold:',pearsonr(df.SalePrice, df.YrSold))

print(sm.OLS(df.YrSold, df.SalePrice).fit().summary())

chart =sns.lmplot(y= 'SalePrice', x='YrSold', data=df)
print('correlation in betwn saleprice and GarageArea:',pearsonr(df.SalePrice, df.GarageArea))

print(sm.OLS(df.GarageArea, df.SalePrice).fit().summary())

chart =sns.lmplot(y= 'SalePrice', x='GarageArea', data=df)
print('correlation in betwn saleprice and ScreenProch:',pearsonr(df.SalePrice, df.ScreenPorch))

print(sm.OLS(df.ScreenPorch, df.SalePrice).fit().summary())

chart =sns.lmplot(y= 'SalePrice', x='ScreenPorch', data=df)
#work with categorical features

# Columns containing text values (dtypes == 'object') are categorical features.
catdf = (df.dtypes == 'object')
cat = list(catdf[catdf].index)

mancat = ['MSSubClass', 'OverallQual', 'OverallCond', ]

c= cat + mancat
data = {}

for i in c:

    v = i

    uniq = len(df[i].unique().tolist())

    data[i] = (v, uniq)
dfcat= pd.DataFrame.from_dict(data, orient='index', columns=['v','uniq'])





ordf = dfcat.sort_values(by = ['uniq'], ascending=True)



plt.figure(figsize=(30,10))

sns.barplot(ordf.v, ordf.uniq)

plt.xticks(rotation=90)

plt.show()

#1

df.Street.unique()
#break it now and plotting# break data into different parts

Paver = df[df.Street == 'Pave']

Gravel = df[df.Street == 'Grvl']

fig, ax=plt.subplots(figsize=(30,10))

sns.distplot(a = np.log(Paver['SalePrice']), label="Paver block", kde=False);

sns.distplot(a = np.log(Gravel['SalePrice']), label="gravel one", kde=False);

plt.legend();
df.BldgType.unique()


singleframe = df[df.BldgType == '1Fam']

doubleframe = df[df.BldgType == '2fmCon']

duplex = df[df.BldgType == 'Duplex']

townhouseeast = df[df.BldgType == 'TwnhsE']

townhousesouth = df[df.BldgType == 'Twnhs']



plt.figure(figsize=(30,10))





sns.distplot(a = np.log(singleframe['SalePrice']), label="single frame", kde=False);

sns.distplot(a = np.log(doubleframe['SalePrice']), label="double frame", kde=False);

sns.distplot(a = np.log(duplex['SalePrice']), label="duplex", kde=False);

sns.distplot(a = np.log(townhouseeast['SalePrice']), label="town house east", kde=False);

sns.distplot(a = np.log(townhousesouth['SalePrice']), label="town house south", kde=False);



plt.legend();
#for house style

plt.figure(figsize=(30,10))

temp = df.groupby(['HouseStyle'],as_index=False)['SalePrice'].median().sort_values(by='SalePrice',ascending=False)

plot1 = sns.boxplot(data=df,x='HouseStyle',y="SalePrice",order=temp['HouseStyle'].to_list());

plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90);
#with exterior

plt.figure(figsize=(30,10))

table = df.groupby(['Exterior2nd'],as_index=False)['SalePrice'].median().sort_values(by='SalePrice',ascending=False)

plot1  = sns.stripplot(data=df,x='Exterior2nd',y="SalePrice",order=table['Exterior2nd'].to_list());

plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90);
# with neighborhood

plt.figure(figsize=(30,10))

temp = df.groupby(['Neighborhood'],as_index=False)['SalePrice'].median().sort_values(by='SalePrice',ascending=False)

plot1  = sns.violinplot(data=df,x='Neighborhood',y="SalePrice",order=temp['Neighborhood'].to_list());

plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90);
#with foundation

plt.figure(figsize=(30,10))

temp = df.groupby(['Foundation'],as_index=False)['SalePrice'].median().sort_values(by='SalePrice',ascending=False)

plot1  = sns.barplot(data=df,x='Foundation',y="SalePrice",order=temp['Foundation'].to_list());

plot1 .set_xticklabels(plot1.get_xticklabels(), rotation=90);
# with lot shape

plt.figure(figsize=(30,10))

temp = df.groupby(['LotShape'],as_index=False)['SalePrice'].median().sort_values(by='SalePrice',ascending=False)

plot1 = sns.boxenplot(data=df,x='LotShape',y="SalePrice",order=temp['LotShape'].to_list());

plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90);
temp = df.groupby(['Neighborhood'],as_index=False)['SalePrice'].median()

temp = temp.sort_values(by='SalePrice',ascending=False)

temp.style.background_gradient(cmap='Blues')
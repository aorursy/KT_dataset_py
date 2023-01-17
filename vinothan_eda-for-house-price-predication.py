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
## import the dataset 
import os
print(os.listdir("../input"))

df=pd.read_csv("../input/train.csv")
df.head(10)
df.shape
df.columns
### givies you the summary
df.describe()
df.info()
df.columns[df.isnull().any()].tolist()
df.isnull().sum().sort_values(ascending=False).head(20)
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data["Percent"],color="green")
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
df.drop(['PoolQC','MiscFeature','Alley','Fence'] ,axis=1, inplace=True)
df.head()
df.isnull().sum().sort_values(ascending=False).head(20)
# syntax to know which colunm has the categorical features
categorical_features = df.select_dtypes(include = ["object"]).columns
categorical_features
# syntax to know which colunm has the numerical features
numerical_features = df.select_dtypes(exclude = ["object"]).columns
numerical_features
# Differentiate numerical features (minus the target) and categorical features
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
f, ax = plt.subplots(figsize=(10, 8))
sns.set_style('whitegrid')
sns.distplot(df.SalePrice,color='red')
plt.xlabel('SalePrice', fontsize=15)
#plt.title('Given Data', fontsize=15)
f, ax = plt.subplots(figsize=(10, 8))
#plt.figure()
#plt.subplot(212)
sns.set_style('whitegrid')
stats.probplot(df['SalePrice'], plot=plt)
plt.xlabel('SalePrice', fontsize=15)
plt.title('Given Data', fontsize=15)
plt.show()
f, ax = plt.subplots(figsize=(12, 10))
plt.figure(1)
plt.subplot(211)
sns.set_style('whitegrid')
sns.distplot(df['SalePrice'].apply(np.sqrt),color='red')
plt.xlabel('Square of SalePrice', fontsize=15)
plt.title('Square Transform', fontsize=15)



f, ax = plt.subplots(figsize=(12, 10))
plt.figure(2)
plt.subplot(212)
sns.set_style('whitegrid')
sns.distplot(df['SalePrice'].apply(np.log),color='red')
plt.xlabel('Log of SalePrice', fontsize=15)
plt.title('Log Transform', fontsize=15)
plt.show()

f, ax = plt.subplots(figsize=(12, 12))
plt.figure(1)
plt.subplot(211)
sns.set_style('whitegrid')
stats.probplot(df['SalePrice'].apply(np.sqrt), plot=plt)
plt.xlabel('Square of SalePrice', fontsize=15)
plt.title('Square Transform', fontsize=15)



f, ax = plt.subplots(figsize=(12, 12))
plt.figure(2)
plt.subplot(212)
sns.set_style('whitegrid')
stats.probplot(df['SalePrice'].apply(np.log), plot=plt)
plt.xlabel('Log of SalePrice', fontsize=15)
plt.title('Log Transform', fontsize=15)
plt.show()

data = pd.concat([df['SalePrice'], df['BedroomAbvGr']],axis=1)
f, ax = plt.subplots(figsize=(10, 8))
fig = sns.boxplot(x='BedroomAbvGr', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=500000);
## plot to check the total number of rooms
data = pd.concat([df['SalePrice'], df['TotRmsAbvGrd']],axis=1)
f, ax = plt.subplots(figsize=(10, 8))
fig = sns.boxplot(x='TotRmsAbvGrd', y="SalePrice", data=data)
#fig.axis(ymin=0, ymax=500000);
# to check total SaleCondition
data = pd.concat([df['SalePrice'], df['SaleCondition']],axis=1)
f, ax = plt.subplots(figsize=(10, 8))
fig = sns.violinplot(x='SaleCondition', y="SalePrice", data=data)
#fig.axis(ymin=0, ymax=500000);
## to check over all conditions of house
data = pd.concat([df['SalePrice'], df['OverallCond']],axis=1)
f, ax = plt.subplots(figsize=(10, 8))
fig = sns.violinplot(x='OverallCond', y="SalePrice", data=data)
### Draw a set of vertical bars with nested grouping by a two variables

## to check over all conditions of house with bed room
data = pd.concat([df['SalePrice'], df['OverallCond'],df['BedroomAbvGr']],axis=1)
f, ax = plt.subplots(figsize=(15, 12))
sns.barplot(x="OverallCond", y="SalePrice", hue="BedroomAbvGr",data=data)
## to check over all conditions of house with sale condintion
data = pd.concat([df['SalePrice'], df['OverallCond'],df['SaleCondition']],axis=1)
f, ax = plt.subplots(figsize=(15, 12))
sns.barplot(x="OverallCond", y="SalePrice", hue="SaleCondition",data=data)
## to check BedroomAbvGr house with sale condintion
data = pd.concat([df['SalePrice'], df['BedroomAbvGr'],df['SaleCondition']],axis=1)
f, ax = plt.subplots(figsize=(15, 12))
sns.boxplot(x="BedroomAbvGr", y="SalePrice", hue="SaleCondition",data=data)
fig.axis(ymin=0, ymax=500000)
#scatter plot grlivarea/saleprice
#var = 'GrLivArea'
data = pd.concat([df['SalePrice'], df['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice',ylim=(0,800000))

#scatter plot TotalBsmtSF /saleprice
#var = 'GrLivArea'
data = pd.concat([df['SalePrice'], df['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice',ylim=(0,800000))
#Deleting outliers
df = df.drop(df[(df['TotalBsmtSF']>2000) & (df['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(df['TotalBsmtSF'], df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()
#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(18, 12))
sns.heatmap(corrmat, vmax=.9, square=True);
# most correlated features
corrmat = df.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="Oranges")
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df[cols], size = 2.5)
plt.show();

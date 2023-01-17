#Inviting Party People

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))



#Load datasets for demonstrations

titanic_data = pd.read_csv("../input/titanic/train.csv")

house_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
#look at first 5 rows using .head()

house_data.head()



#Wanna see more?. try -> house_data.head(13) for first 13 rows.
#look at last 5 rows using .tail()

house_data.tail()
house_data.shape
house_data.columns
#Peek... head or tail

house_data['SalePrice'].head()
# Descriptive statistics summary

house_data['SalePrice'].describe()
from scipy.stats import norm

# Distribution plot

def distribution_plot(data):

    sns.distplot(data, fit=norm)

    plt.ylabel('Frequency')

    plt.title(f'{data.name} distribution')

    

distribution_plot(house_data['SalePrice'])
#skewness and kurtosis

print("Skewness: %f" % house_data['SalePrice'].skew())

print("Kurtosis: %f" % house_data['SalePrice'].kurt())
OverallQual = house_data['OverallQual'].astype('category')



#Peek... head or tail

OverallQual.head()
# Descriptive statistics summary

OverallQual.describe()
column = OverallQual;

print('Column Name:{}\nCardinality:{}\nValues:{}'.format(column.name,column.nunique(), column.unique()))
OverallQual.value_counts()
def getPlotsforCatFeature(series,figX=15,figY=7):

    f,ax=plt.subplots(1,2,figsize=(figX,figY))

    series.value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0])

    ax[0].set_title(f'{series.name}')

    ax[0].set_ylabel('')

    sns.countplot(series,ax=ax[1])

    ax[1].set_title(f'Count plot for {series.name}')

    plt.show()

    

getPlotsforCatFeature(OverallQual,15,5)
#scatter plot

house_data.plot.scatter(x='GrLivArea', y='SalePrice');



''' Alternatively you could use following function 

def scatterplot(seriesX,seriesY):

    data = pd.concat([seriesY, seriesX], axis=1)

    data.plot.scatter(x=seriesX.name, y=seriesY.name)

    

scatterplot(house_data['GrLivArea'],house_data['SalePrice'])

'''
#Box plot

num = 'SalePrice'

cat = 'OverallQual'

df  =  house_data



data = pd.concat([df[num], df[cat]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=cat, y=num, data=data)

fig.axis(ymin=0, ymax=800000);
def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)



def fillMissingCatColumns(data,categorical):

    for c in categorical:

        data[c] = data[c].astype('category')

        if data[c].isnull().any():

            data[c] = data[c].cat.add_categories(['MISSING'])

            data[c] = data[c].fillna('MISSING')

    

def getboxPlots(data,var,categorical):

    fillMissingCatColumns(data,categorical)

    f = pd.melt(data, id_vars=var, value_vars=categorical)

    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

    g = g.map(boxplot, "value", var)

    



data = house_data.copy()

categorical = [f for f in data.columns if data.dtypes[f] == 'object']    

getboxPlots(data,'SalePrice',categorical)
def getCorrHeatMap(dataFrame,figSize=[12,9]):

    corrmat = dataFrame.corr()

    f, ax = plt.subplots(figsize=(figSize[0], figSize[1]))

    sns.heatmap(corrmat, vmax=.8, square=True);



getCorrHeatMap(house_data)
def getZoomedCorrHeatMap(dataFrame,featureCount,target,figSize=[12,9]):

    corrmat = dataFrame.corr()

    cols = corrmat.nlargest(featureCount, target)[target].index

    f , ax = plt.subplots(figsize = (figSize[0],figSize[1]))

    cm = np.corrcoef(dataFrame[cols].values.T)

    sns.set(font_scale=1.25)

    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

    plt.show()



getZoomedCorrHeatMap(house_data,10,'SalePrice',[10,8])
def getMissingValuesInfo(df):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100, 2)

    temp = pd.concat([total, percent], axis = 1,keys= ['Total Missing Count', '% of Total Observations'])

    temp.index.name ='Feature Name'

    return temp.loc[(temp['Total Missing Count'] > 0)]



getMissingValuesInfo(house_data)
# Visualizing missing counts

missing = house_data.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

plt.subplots(figsize=(15,5))

missing.plot.bar()

plt.show()
fig, ax = plt.subplots(figsize=(20,5))

sns.heatmap(house_data.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
def distplots(data,num_features):

    f = pd.melt(data, value_vars=num_features)

    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)

    g = g.map(sns.distplot, "value")

    



num_features = house_data.select_dtypes(include=['int64','float64'])

distplots(house_data,num_features)
num_features = house_data.select_dtypes(include=['int64','float64'])

num_features.describe()
categorical_features = house_data.select_dtypes(include='object')

categorical_features.describe()
def printUniqueValues(df,cardinality=1000):

    n = df.select_dtypes(include=object)

    for column in n.columns:

        uCount = df[column].nunique()

        if uCount<=cardinality:

            print('{:>12}: {} {}'.format(column,uCount, df[column].unique()))

            #print(column,': [',uCount , '] ', df[column].unique())





printUniqueValues(house_data,10)
import pandas_profiling

profile_report = pandas_profiling.ProfileReport(titanic_data)

#profile_report.to_file("profile_report.html")

profile_report
# We can use pandas profiling on selected features too.



# Using Pandas Profiling to analyse SalePrice feature in housing dataset.

import pandas_profiling

series = house_data['SalePrice']

d = { series.name : series}

df = pd.DataFrame(d) 

pandas_profiling.ProfileReport(df)
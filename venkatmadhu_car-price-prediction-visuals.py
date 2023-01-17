#Data wrangling: 

import numpy as np 

import pandas as pd



#Visualisation :

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('fivethirtyeight')



#View max columns:

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
data = pd.read_csv('../input/automobile-dataset/Automobile_data.csv')

data.head(1)
print('shape of the data:',data.shape)

print(" ")

print("*"*10)

print('Unique brands:',len(data.make.unique()),data.make.unique())

print(" ")

print("*"*10)

print('Fuel type:',data['fuel-type'].unique())

print(" ")

print("*"*10)

print(data.info())

print(" ")

print("*"*10)

print(data.describe())

print(" ")

print("*"*10)

print('Missing Values:')

print(data.isnull().sum().sort_values(ascending=False))
data.head(1)
print(data['num-of-doors'].value_counts())
data.loc[data['num-of-doors']=='?','num-of-doors'] = 'four'

data.loc[data['num-of-cylinders']=='?','num-of-cylinders'] = 'four'
def replaceMissingValue(featureName):

    data.loc[data[featureName]=='?',featureName] = data.loc[data[featureName]!='?',featureName].median()

    data[featureName] = data[featureName].astype(float)

    return data
data = replaceMissingValue('normalized-losses')

data = replaceMissingValue('horsepower')

data = replaceMissingValue('bore')

data = replaceMissingValue('stroke')

data = replaceMissingValue('peak-rpm')



data['symboling'] = data['symboling'].astype(int)



data = data[data.price!='?']

data['price'] = data['price'].astype(int)

print('Shape of the data',data.shape)
data.head(1)
f,ax=plt.subplots(1,2,figsize=(22,8))



sns.countplot(y = 'make',order = data['make'].value_counts().index,data=data)

ax[0].set_title('Distribution of Brands')





data['make'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[1].set_title('Distribution of Brands')

f,ax=plt.subplots(1,2,figsize=(18,8))



sns.countplot('fuel-type',order = data['fuel-type'].value_counts().index,data=data,ax=ax[0])

ax[0].set_title('Fuel Type')



sns.boxplot(x=data['fuel-type'],y = data['price'],data=data,ax=ax[1])

ax[1].set_title('FuelType Vs Price')
f,ax=plt.subplots(1,2,figsize=(20,8))



data['body-style'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Type of Vehicle')



sns.boxplot(x=data['body-style'],y = data['price'],data=data,ax=ax[1])

ax[1].set_title('body-type Vs Price')
data.head(1)
f,ax=plt.subplots(1,2,figsize=(20,8))



data['num-of-cylinders'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Count of Cylinders')



sns.boxplot(x=data['engine-type'],y = data['horsepower'],data=data,ax=ax[1])

ax[1].set_title('Enginetype Vs Horsepower')
f,ax = plt.subplots(1,2,figsize=(20,8))



data['avg_mileage'] = (data['city-mpg'] + data['highway-mpg'])/2

sns.scatterplot(x='horsepower',y = 'avg_mileage' ,data = data,ax=ax[0])

ax[0].set_title('Horsepower Vs Mileage')



sns.scatterplot(x='engine-size',y = 'avg_mileage' ,data = data,ax=ax[1])

ax[1].set_title('engine-size Vs Mileage')
fig = plt.figure(figsize=(15, 10))

mileage=data.groupby(['make']).mean()

mileage['avg-mpg']=((mileage['city-mpg']+mileage['highway-mpg'])/2)

ax=mileage['avg-mpg'].sort_values(ascending=False).plot.bar(edgecolor='k',linewidth=2)

plt.xticks(rotation='vertical')

plt.xlabel('Car Maker',fontsize=20)

plt.ylabel('Number of cars',fontsize=20)

plt.title('Fuel Economy of Car Makers',fontsize=30)

ax.tick_params(labelsize=20)

plt.show()
plt.figure(figsize=(10,8))

sns.boxplot(x='num-of-doors',y = 'price' ,data = data)

plt.title('Two/Four Seater car Vs Price')
plt.rcParams['figure.figsize']=(23,15)

ax=sns.factorplot(data=data, x="num-of-cylinders", y="horsepower");

plt.rcParams['figure.figsize']=(23,10)

ax=sns.boxplot(x='drive-wheels',y='price',data=data,width=0.8,linewidth=5)

ax.set_xlabel('Make of Car',fontsize=30)

ax.set_ylabel('Price in $',fontsize=30)

plt.title('Price of Car Based on Make',fontsize=40)

ax.tick_params(axis='x',labelsize=20,rotation=90)
plt.figure(figsize=(20,10))

sns.heatmap(data.corr(),annot=True,cmap='summer');
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sbn
train = pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')

nas = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])

predCol = train['SalePrice']

train.drop('SalePrice',axis=1,inplace=True)

data = pd.concat([train,test],axis=0,ignore_index=True)
#Missing Values

def missingValues(dataCols):

    for col in dataCols:

        if data.ix[data[col].isnull(),'Id'].count() > 0:

            #print(str(i)+" "+str(data.ix[data[i].isnull(),'Id'].count()))

            if data.ix[data[col].isnull(),'Id'].count() >= 0.5*len(data):

                data.drop(col,axis=1,inplace=True)

            elif str(data[col].dtype) !='object' and str(data[col].dtype) !='category':     

                data[col] = data[col].fillna(data[col].mean())

            elif str(data[col].dtype) =='object' or str(data[col].dtype) !='category':

                data[col] = data[col].astype("category")

                data[col] = data[col].fillna(data[col].mode()[0])

    return data 
data = missingValues(data.columns[1:])

#Checking for wheather any missing values stil presists

for col in data:

        if data.ix[data[col].isnull(),'Id'].count() > 0:

             print(str(col)+" "+ str(data[col].dtype) +" "+str(data.ix[data[col].isnull(),'Id'].count()))

predCol = np.log(predCol)

sbn.distplot(predCol)
data.head(5)
cat =[]

for col in data:

    if len(data[col].value_counts()) <= 15:

        cat.append(str(col))

cat.extend(['MSSubClass','Neighborhood','YearBuilt','YearRemodAdd','Exterior2nd'])

print(len(cat))
data[[col for col in data.columns if col not in cat]].head()
data.info()
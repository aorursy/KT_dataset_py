# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
salesdata = pd.read_csv("../input/SalesKaggle3.csv")



#salesdata=salesdata[salesdata['ReleaseYear']>0]
salesdata.columns
salesdata.shape

#There is one record whichin which the year is labelled as "0" and it is a active record,

#it may be done by mistake,letsexamine the sku number and find out the release year

salesdata.loc[salesdata['ReleaseYear'] == 0]

#salesdata=salesdata[salesdata['ReleaseYear']>0]
salesdata[salesdata['SKU_number']== 294185]

#the skn number has two records,one active and one historical,historical year is mentioned as 2008

#we can delete the two rows as ,sales count and sales flag is not available and also ,there are only two records,there is not much information we are losing

salesdata = salesdata[salesdata['SKU_number'] != 294185]
salesdata.head(5)
#percentage of records which has zerosales,by identifing we can discontinue the manufacturing of those products

#nearly 30  percentage of total products have no sales

(salesdata[salesdata['SoldCount'] == 0 ].shape[0]/salesdata.shape[0])*100
#we dont have any information about soldcount and sold flag
sknwiseitemcount = pd.pivot_table(salesdata,values='ItemCount', index='SKU_number', columns=['File_Type'], 

                                            fill_value=None, margins=False, dropna=True)

sknwiseitemcount.head(5)
sknwiseitemcount.fillna(0)

sknwiseitemcount['increase'] = (sknwiseitemcount.Active - sknwiseitemcount.Historical)

increasedcount= pd.DataFrame(sknwiseitemcount[sknwiseitemcount['increase']>0]['increase'])

print(increasedcount[increasedcount['increase'] == max(increasedcount['increase'])])

print(increasedcount[increasedcount['increase'] == min(increasedcount['increase'])].count())

##SKU_numbe 622205  has more itemcount than historical records

##there are 3799 products which just icreased its production to one unit

#there are a 22607 products withmore than one increase in itemcount 

# fig, ax = plt.subplots(figsize=(20,20))

# increasedcount.plot.bar(ax =ax)

print(increasedcount[increasedcount['increase'] >1].count())
fig, ax = plt.subplots(figsize=(20,20))





(pd.pivot_table(salesdata,values='SoldCount', index='ReleaseYear', aggfunc='sum')).plot.bar(title = "Freq dist of Soldcount on year basis",ax = ax)
fig, ax = plt.subplots(figsize=(15,15))

(pd.pivot_table(salesdata,values='SoldCount', index='SKU_number', aggfunc='sum')

).plot.bar(title = "Freq dist of Soldcount on SKUNumber basis",ax = ax)
for i in salesdata.columns:

    print("unique values in {} {}".format(i,len(pd.unique(salesdata[i]))))
for i in salesdata.columns:

    print("NA values in {} is :{}".format(i, salesdata[i].isnull().sum()))
#Soldflag and sold count are missing

#Percentage missing values 

#61 percent of NA vales are there in SalesCount and Sales

print(salesdata['SoldCount'].isnull().sum()/len(salesdata.SoldCount))
#plt.plot(salesdata.groupby('ReleaseYear')['SoldCount'].count())

#plt.show()

fig, ax = plt.subplots(figsize=(15,7))

salesdata.groupby('ReleaseYear')['SoldCount'].count().plot(ax=ax)
salesdata.groupby(['ReleaseYear','SoldCount'])['SoldCount'].sum()
numbertype = []

for i in salesdata.columns:

    #print(salesdata[i].dtype)

    if salesdata[i].dtype != "object":

        numbertype.append(i)

print(numbertype)

numbertype.remove('Order')

numbertype.remove('SKU_number')

print(numbertype)



numericaldata = salesdata[numbertype]
fig, ax = plt.subplots(figsize=(15,15))

numericaldata.hist(ax = ax)

plt.show()

fig, ax = plt.subplots(figsize=(15,15))

numericaldata.plot(kind='box', subplots=True,ax= ax)

plt.show()
import numpy 

fig, ax = plt.subplots(figsize=(15,15))

correlations = numericaldata.corr()

# plot correlation matrix



cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = numpy.arange(0,10,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(numbertype)

ax.set_yticklabels(numbertype)

plt.show()
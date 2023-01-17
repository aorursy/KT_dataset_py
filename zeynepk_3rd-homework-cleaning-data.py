# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/metal_bands_2017.csv",encoding = "ISO-8859-1")
data.head(10) #1-10 columns
#we can see whats mostly data about, also number of columns
data.shape #(number of rows,number of rows' values)
data.info() #we have 8 null value in origin
print(data.origin.value_counts()) #frequency of origin
#for example 1139 bands have origin in USA
data.describe() #also split and formed columns contain number but they are object
avrg_fan = sum(data.fans)/len(data.fans)
data["new_title"]=[True if i > avrg_fan else False for i in data.fans]
data.new_title           #i made this for making boxplot of fan number
data.boxplot(column="fans", by="new_title")
#so here it is, 5000 metal bands have fans more than average to the dataset
#tidying data with melt()
#melt provides us seeing dataset more "tidy" or easy
data10 = data.head(10) #taking top 10 of data
data10
melteddata = pd.melt(frame=data10,id_vars="band_name",value_vars=["fans"])
melteddata
melteddata.pivot(index = 'band_name', columns = 'variable',values='value')
#reverse of melt()
#concatenating data
data1 = data.tail(3) #last 3
data2 = data.head(3) #top 3
concted_data = pd.concat([data1,data2],axis=0,ignore_index=True)
concted_data                           #vertical |
data3 = data['fans'].head()    
data4 = data['band_name'].head()
concted_data2 = pd.concat([data3,data4],axis =1) #horizontal -
concted_data2
#data types and converting them 
print(data.origin.dtype, "\n", data.fans.dtype) 
data["origin"] = data["origin"].astype("category")    #origin converted to category from object
data["fans"] = data["fans"].astype("float64")         #fans converted to float from integer
print(data.origin.dtype, "\n", data.fans.dtype)     
## missing data
data.info()
#there are 8 null values
data["origin"].dropna(inplace = True) #droped the NaNs , now i'll check with assert if it worked 
assert data["origin"].notnull().all() #turns nothing, so it is true
#another way
data["fans"].fillna('empty',inplace = True)
assert data["fans"].notnull().all()          #there are no null in fans already
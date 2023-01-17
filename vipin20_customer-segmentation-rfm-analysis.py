# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing requied libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from mlxtend.frequent_patterns import apriori #, association_rule
#Loading the data
#may be there is non standerd null values, replace all as NaN
df=pd.read_csv("/kaggle/input/transaction-data/transaction_data.csv",na_values=(" ","?","_","-1"))
df.head()
#disply the bottom 5 rows
df.tail()
#explore total number of rows and columns
df.shape

df=df[pd.notnull(df["UserId"])]
#dropingduplicates record because a huge data set as in thes case contains 1083818 recodes
#often have some duplicate data which might be disturbing.
df=df.drop_duplicates()
df.head()
#checking the shape after removing duplicate values
df.shape
#explore columns of the data for forther use
df.columns
#removing spaces from item discription
df['ItemDescription']=df['ItemDescription'].str.strip()
#renaming the columns
df=df.rename(columns={"NumberOfItemsPurchased":"Quantity"})
#customers ordered negative quantity,which is not possible.so we filter quantity grater then zero
df=df[(df["Quantity"]>0)]
#finding the null values
df.isnull().sum()
df.info()
#droping the missing values
df=df.dropna()
df.count()
#adding a new column total Cost
df['Total_Cost']=df['Quantity']*df['CostPerItem']
df.head()
#finding those coustomers who done max shoping.
s_data=df.query('Country=="United Kingdom"').sort_values("Total_Cost",ascending=False)
s_data.head()
#finding most expensive items
most_exp_items=df.sort_values("CostPerItem",ascending=False)
most_exp_items.head()

#finding max and min for each column
df.agg(["max","min"])

df['Country'].value_counts()
df.Country.value_counts().nlargest(15).plot(kind="bar",figsize=(10,5))
plt.title("Number of Country")
plt.ylabel("number of Country")
plt.xlabel("country")
uk_data=df[df.Country== "United Kingdom"]
uk_data.head()
uk_data.describe()
#finding time when min and max transaction done
uk_data['TransactionTime'].min(),uk_data['TransactionTime'].max()
PRESENT=dt.datetime(2018,7,26)
uk_data['TransactionTime'] = pd.to_datetime(uk_data['TransactionTime'])
data=uk_data[["UserId","TransactionId","TransactionTime","Quantity","CostPerItem","Total_Cost"]]
data.head()
rfm= uk_data.groupby('UserId').agg({'TransactionTime': lambda date: (PRESENT - date.max()).days,
                                    'TransactionId': lambda num: len(num),
                                    'Total_Cost': lambda price: price.sum()})
rfm.columns
# Change the name of columns 
rfm.columns=['monetary','recency','frequency']
rfm['recency'] = rfm['recency'].astype(int)
rfm.head()
rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, ['1','2','3','4']) 
rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4','3','2','1']) 
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1'])
rfm.head()
rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
rfm.head()
rfm.tail()


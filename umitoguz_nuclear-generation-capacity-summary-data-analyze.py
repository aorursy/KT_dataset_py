# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



data = pd.read_csv('../input/capacity-summary-data/MER_T08_01.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(15)




data.columns



data.plot(kind='scatter', x='Value', y='Column_Order',alpha = 0.5,color = 'blue')

plt.xlabel('Value')              # label = name of label

plt.ylabel('Column_Order')

plt.title('Latitude Longitude Scatter Plot')
x=data['Value']==10

data[x]
for index,value in data[['Value']][0:100].iterrows():

    print(index," : ",value)
i = 0

while i != 10.0 :

    print('Value is: ',i)

    i +=1.0 

print(i,' is equal to 10.0')
data.tail()
data.shape
print(data['Unit'].value_counts(dropna =False))
data.describe()




data.boxplot(column='Column_Order',by = 'Value')







data_new = data.head()    # I only take 5 rows into new data

data_new



data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row







data1 = data['Value'].head()

data2= data['Column_Order'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col



data.dtypes
data["Value"].value_counts(dropna =False)
data1 = data.loc[:,["Value","YYYYMM","Column_Order"]]

data1.plot()
data1.plot(subplots = True)
data1.plot(kind = "scatter",x="Value",y = "Column_Order")
data.describe()
data["Value"][1]
data.Value[1]


data[["Value","Column_Order"]]



print(type(data["Value"]))    

data.loc[1:10,"Value":"Column_Order"] 
data.loc[10:1:-1,"Value":"Column_Order"] 
boolean = data.Value> 9

data[boolean]
first_filter = data.Value > 2.5

second_filter = data.Column_Order > 1.36

data[first_filter & second_filter]




data.Value[data.Column_Order>3.35]



print(data.index.name)

# lets change it

data.index.name = "index_name"

data.head()
data.pivot(columns ="Value",values="Column_Order")
data1 = data.set_index(["Value","Column_Order"])

data1
data.groupby("Value").mean() 




data.groupby("Value").Column_Order.max() 



data.groupby("Value")[["Column_Order","YYYYMM"]].min() 
plt.plot([data['Value'],data['Column_Order']])

plt.xlabel('Value')

plt.ylabel('Column_Order')

plt.show()
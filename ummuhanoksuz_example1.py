# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data.head()
data.tail()
data.columns

data.shape
data.info()

# We learn something our datas.
print(data['neighbourhood_group'].value_counts(dropna=False))
print(data.minimum_nights.mean)
data.describe()
#data.boxplot(column='room_type',by='price')

#plt.show()

#ANLAMADIMMMMM
data_new=data.head()

print(data_new)
melted=pd.melt(frame=data_new,id_vars='name',value_vars=['host_name','neighbourhood_group'])

melted
melted=pd.melt(frame=data_new,id_vars='name',value_vars=['neighbourhood_group','neighbourhood'])

melted
melted.pivot(index='name',columns='variable',values='value')
data1=data.head()

data2=data.tail()

conc_data_row=pd.concat([data1,data2],

                        axis=0,

                       ignore_index=True)

conc_data_row
data3=data['neighbourhood_group'].head()

data4=data['host_name'].tail()

conc_data_col=pd.concat([data3,data4],

                        axis=1)

conc_data_col
data.dtypes
#We can change data type

data['name']=data['name'].astype('category')

data.dtypes
data['name']=data['name'].astype('object')

data.dtypes
data.info()
data['minimum_nights'].value_counts(dropna=False)
data['minimum_nights'].dropna(inplace=True)

assert 1==1 #We check in

assert data['minimum_nights'].notnull().all()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head()
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
#Line Plot

#color = color, label = label, linewidth = width of line, alpha= opacity, frid= grid, linestyl e = sytle of line

data.price.plot(kind = 'line', color= 'g', label= 'price',linewidth =1, alpha=0.5 , grid= True,linestyle =':')

data.number_of_reviews.plot(color ='r', label= 'number_of_reviews' ,linewidth =1, alpha= 0.5, grid= True, linestyle= '-.')

plt.legend(loc='upper right')  #legend = puts label into plot

plt.xlabel= ('x axis')         #label = name of label

plt.ylabel= ('y axis')

plt.title = ('Line Plot')    #title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.availability_365.plot(kind= 'hist', bins = 50, figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.price.plot(kind= 'hist' , bins=50 )

plt.clf()

# We cannot see plot due to clf()
# 1 - Filtering Pandas data frame

x = data['availability_365']>194    # A lots are availability

data[x]
# 2 - Filtering Pandas with logical_and

# There are new york city airbnb who have higher availability_365 value than 194 and higher price value than 150

data[np.logical_and(data['availability_365']>194, data['price']>150)]
#This is also same with previous code line. ThereFore we can also use '&' for filtering

data[(data['availability_365'] > 194) & (data['price'] > 150)]
# Stay in loop if condition( i is not equal 250) is true

price = 0

while price != 249 :

    print('price is: ',price)

    price +=1

    print(price,' is smaller than 250')

data.head()
#average price = total price / RangeIndex

price_average = sum(data.price)/ 48895

price_average

data["price_level"] =["expensive" if i > price_average else "cheap" for i in data.price]

data.loc[:15,["price_level", "price"]]
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head() # head shows first 5 rows
# tail shows last 5 rows

data.tail()
# columns gives column names of features

data.columns
# shape gives number of rows and columns in a tuble

data.shape 
# info gives data of informations

data.info()
# how much Nan informations? 

print(data['last_review'].value_counts(dropna=False))
data.describe()
# For example: compare hotels of price that are availability 365 or not

# Black line at top is max

# Blue line at top is 75%

# Green line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min



# Datas are not good. You see.



data.boxplot(column='price', by ='availability_365')
data_new = data.head()    # I only take 5 rows into new data

data_new
# Melt example

# id_vars = what we do not wish to melt

# value_vars = what we want to melt



melted = pd.melt(frame =data_new, id_vars = 'name' , value_vars = ['neighbourhood_group', 'room_type'])

melted
# Pivot Table

melted.pivot(index= 'name' , columns = 'variable', values='value')
#Concatenating Data (concat)

data1 = data.head()

data2 = data.tail()



conc_data_row = pd.concat ([data1,data2], axis=0 , ignore_index = True) # axis=0 : adds dataframes in row

conc_data_row

# Other Example -- Concatenating Data

data1= data['neighbourhood_group'].head()

data2 = data['room_type'].head()

conc_data_col = pd.concat ([data1,data2],axis= 1) # axis = 1 : adds dataframes in columns

conc_data_col
#Data types

data.dtypes
# Lets convert object(str) to categorical and int to float

data['host_name'] = data ['host_name'].astype('category')

data['price'] = data ['price'].astype('float')
data.dtypes
data['host_name'] = data ['host_name'].astype('object')

data['price'] = data ['price'].astype('int')
data.dtypes
# Missing Data and Testing with Assert

# How many NaN data?

data.info()
# Lets check last Review

data["last_review"].value_counts(dropna = False)

# You see , there are 10052 NaN value
# Lets drop NaN value

data1 = data

data1["last_review"].dropna(inplace= True)

#does it work?
# Lets chechk with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true
# In order to run all code, we need to make this line comment

# assert 1==2 # return error because it is false
assert  data['last_review'].notnull().all() # returns nothing because we drop nan values
data["last_review"].fillna('empty',inplace = True)
assert  data['last_review'].notnull().all() # returns nothing because we do not have nan values
# # With assert statement we can check a lot of thing. For example

# assert data.columns[1] == 'Name'

# assert data.last_review.dtypes == np.int
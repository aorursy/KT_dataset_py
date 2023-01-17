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
import unicodedata

data=pd.read_csv('../input/superstore-data/superstore_dataset2011-2015.csv',encoding = "ISO-8859-1")
data.head()
data.info()
data.describe()
data.shape
data.columns
import seaborn as sns

from matplotlib import pyplot  as plt

data.Region.value_counts().plot(kind='bar')

plt.title('counts across all regions')

plt.ylabel('Counts')

plt.xlabel('region distribution')
data.Market.value_counts().plot(kind='bar')
data.Category.value_counts().plot(kind='pie')
pd.crosstab(data['Quantity'].count(),data.Category).plot(kind='bar')

plt.ylabel('Counts')

plt.xlabel('quantity')

plt.title('category vs quantity')

cont = pd.get_dummies(data['Product Name'],prefix='Product Name',drop_first=True)

#Adding the results to the master dataframe

data2 = pd.concat([data,cont],axis=1)
col = list(data.columns)

del col[1]
for c in col:

    del data2[c]
c1 = list(data2.columns)

del c1[0]
d = data2.groupby(data2['Order ID']).sum()  
orders = pd.Series(data2['Order ID'].unique())

orders
#Here we will consider only those orders where the basket size is greater than 2. 

#This is done to reduce computational load.

d_t = d.T
f=[]

for col in orders:

    if d_t[col].sum()>2:

        f.append(col)
t=pd.DataFrame()
t['Order ID']=f
datafinal = d.merge(t,on='Order ID',how='inner')
#Here we will only consider those items where the count of its sale is greater than 20

product_support_dict = []

for column in c1:

    if sum(datafinal[column])<=20:

        product_support_dict.append(column)
for col in product_support_dict:

    del datafinal[col]
orders = datafinal.index.values

products = datafinal.columns.values
orders = list(orders)

products = list(products)

datafinal.set_index('Order ID',inplace=True)

datafinal
transaction_matrix = datafinal.as_matrix()

# get number of rows and columns

rows, columns = transaction_matrix.shape

# init new matrix

frequent_items_matrix = np.zeros((datafinal.shape[1],datafinal.shape[1]))

# compare every product with every other

for this_column in range(0, columns):

    for next_column in range(this_column + 1, columns):

        # multiply product pair vectors

        product_vector = transaction_matrix[:,this_column] * transaction_matrix[:,next_column]

        # check the number of pair occurrences in baskets

        count_matches = sum((product_vector)>0)

        # save values to new matrix

        frequent_items_matrix[this_column,next_column] = count_matches
plt.imshow(frequent_items_matrix)

plt.colorbar()

plt.show()
# and finally combine product names with data

frequent_items_df = pd.DataFrame(frequent_items_matrix, columns = datafinal.columns.values, index = datafinal.columns.values)

 

import seaborn as sns

# and plot

sns.heatmap(frequent_items_df)
%matplotlib inline

sns.heatmap(frequent_items_df)
# extract product pairs with minimum frequency(treshold) basket occurrences

from collections import OrderedDict 

def extract_pairs(treshold,frequent_items_matrix,product_names):

    output = {}

    # select indexes with larger or equal n

    matrix_coord_list = np.where(frequent_items_matrix >= treshold)

    # take values

    row_coords = matrix_coord_list[0]

    column_coords = matrix_coord_list[1]

    # generate pairs

    for index, value in enumerate(row_coords):

        #print index

        row = row_coords[index]

        column = column_coords[index]

        # get product names

        first_product = product_names[row]

        second_product = product_names[column]

        # number of basket matches

        matches = frequent_items_matrix[row,column]

        # put key values into dict

        output[first_product+"-"+second_product] = matches

 

    # return sorted dict

    sorted_output = OrderedDict(sorted(output.items(), key=lambda x: x[1]))

    return sorted_output

 

# plot pairs with minimum frequency of 1 basket matches

pd.Series(extract_pairs(1,frequent_items_matrix,products)).plot(kind="barh")

finallist = pd.Series(extract_pairs(1,frequent_items_matrix,products))
finallist = pd.Series(extract_pairs(1,frequent_items_matrix,products))

finallist 
finallistD = pd.DataFrame()

finallistD['Count']=finallist
finallistD
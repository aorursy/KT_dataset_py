# Let's import necessary libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/iris/Iris.csv')

data.head()  # see first 5 rows
data.tail() # see last 5 rows
data.columns # see variables
# shape gives number of rows and columns in a tuble

data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

data.info()
print(data['Species'].value_counts(dropna =False))  # if there are nan values that also be counted
data.describe()
data.boxplot(column=['SepalLengthCm', 'SepalWidthCm'],by = 'Species',layout=(2, 1), fontsize=10)

data.boxplot(column=['PetalLengthCm','PetalWidthCm'],by = 'Species',layout=(2, 1), fontsize=10)

plt.show()
data_head = data.head()

data_head
data_tail= data.tail()

data_tail
# let's melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted1 = pd.melt(frame=data_head,id_vars = 'Species', value_vars= ['SepalWidthCm','PetalWidthCm'])

melted1
melted2 = pd.melt(frame=data_tail,id_vars = 'Species', value_vars= ['SepalWidthCm','PetalWidthCm'])

melted2
# lets concatenate 2 melted dataframe

conc_data_row = pd.concat([melted1,melted2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data1 = data['Id'].head()

data2= data['Species'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 1 : adds dataframes in column

conc_data_col
data.dtypes
# lets convert object(str) to categorical and float to int.

data['SepalLengthCm'] = data['SepalLengthCm'].astype('int')

data['Species'] = data['Species'].astype('category')

data.dtypes
data.head()
data.info()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/fifa19/data.csv') # import data as dataframe

data.head() # first 5 rows of data

data.tail()
data.columns # shows all columns
data.shape # shows how many rows and columns that data have
data.Position.value_counts(dropna = False) # frequency of positions
# we can use box-plots for detecting mean, max, min and outliers

data.boxplot(column = 'Aggression', by = 'Age', figsize = (18,18))
# melting data

data_new = data.head(10) # create new dataframe

melted = pd.melt(frame = data_new, id_vars = 'Name', value_vars = ['Stamina'])

print(melted)
# concatenating data

data01 = data.head()

data02 = data.tail()

concat_data = pd.concat([data01,data02], axis = 0)

concat_data
data03 = data.Overall.head()

data04 = data.Potential.head()

concat02 = pd.concat([data03,data04], axis = 1)

concat02
# data types

data.dtypes
# converting data types

data.Age = data.Age.astype('float')

data.dtypes
data.columns
data.Club.value_counts(dropna = False)
# drop nan values

data05 = data

data05.Club.dropna(inplace = True)

assert data05.Club.notnull().all() # check if droping is done
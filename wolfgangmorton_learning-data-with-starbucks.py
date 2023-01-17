# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will 

# list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
os.chdir('../input')
os.listdir()
directory = pd.read_csv('directory.csv')
directory.head()
brand = directory['Brand']
brand.str.lower()
brand.str.upper()
brand.str.len()
directory.head()
# String methods on Index are useful for cleaning up/transforming DF cols.

# directory.columns is a Index object that the str accessor. 

idx = pd.Index(directory.columns)
idx
# removes the whitespace

idx.str.strip()
idx.str.lower()
idx.str.upper()
# removes whitespace from both sides.

idx.str.strip()
# removes whitespace from the left.

idx.str.lstrip()
# removes whitespace from the right.

idx.str.rstrip()
idx2 = pd.Index(brand)
brand_cat = brand.astype('category')
brand_cat.str.lower()
# split method returns a Series of lists:

directory.head()
longitude = directory['Longitude']
split_longitude = longitude.astype(str).str.split('.')
split_longitude[:11]
# Elements in split lists can be accessed using get or [] notation.



# Gets the first element in the list.

longitude.astype(str).str.split('.').str.get(0)
longitude.astype(str).str.split('.').str.get(1)
longitude[0]
longitude.astype(str).str.split('.').str[0]
longitude.astype(str).str.split('.').str[1]
longitude.astype(str).str.split('.').str[0:2]
longitude_split = longitude.astype(str).str.split('.')
longitude_split.str[0]
longitude_split.str[1]
# You can easily expand the split list to return a dataframe.

longitude_split.str[0:2]
longitude_string = longitude.astype(str)
df_longitude = longitude_string.str.split('.', expand=True)

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read in dataset

df = pd.read_csv("../input/dsm-beuth-edl-demodata-dirty.csv")
df
# get the number of missing data points per column

missing_values_count = df.isnull().sum()

# look at the # of missing points in the first ten columns

missing_values_count[0:10]
# get percentage of missing data

total_cells = np.product(df.shape)

total_missing = missing_values_count.sum()

(total_missing/total_cells) * 100
#drop empty rows

df = df.dropna(axis=0, how='all')

df
#drop duclicate rows

df = df.drop_duplicates(subset=df.columns.difference(['id']))

df
#reset id column

df.id = range(len(df.id))

df
#set remaining NAs to "unknown"

df = df.fillna(value='unknown')
# drop repetitive column "full_name" 

df = df.drop(['full_name'], axis=1)
#replace "-" in column "age" with nothing and replace "old" with 80 :) and set type to int

df['age'] = df['age'].str.replace('-','')

df['age'] = df['age'].str.replace('old','80')

df['age'] = df['age'].astype('int')

df
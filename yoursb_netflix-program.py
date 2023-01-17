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
# read the csv file

df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
# find out the basic information of the data set

# some columns have many missing values like director

def data_inf(df):

    print('the number of movies and TV shows:', df.shape[0], '\n')

    print('basic information of the data set: ')

    df.info()

    print('\n')

    print('the number of missing values of rach column: ')

    isn = df.isnull().sum()

    print(isn[isn>0])

    df.describe()

data_inf(df)
# show duplicate data

d = df.duplicated(['type', 'title', 'director', 'country', 'release_year'])

df[d]
# drop duplicate data

df = df.drop_duplicates(['type', 'title', 'director', 'country', 'release_year'])
df['date_added'] = df['date_added'].fillna('666')

df[(df.date_added=='666')].index.tolist()
# drop the data without date_added

df = df.drop([6223, 6224, 6225, 6226, 6227, 6228, 6229, 6230, 6231, 6232, 6233])
df['date_added'].isnull().value_counts()
# add the column of added year to the data set

col_name = df.columns.tolist()

print(col_name)

col_name.insert(7, 'year_added')

df = df.reindex(columns=col_name)

a = df['date_added'].str.split(', ').str[-1]

df['year_added'] = a.astype("int")

print(df['year_added'])

print(col_name)

# discover the interval between release date and added date

plt.hist(df['year_added']-df['release_year'])

plt.xlabel('year')

plt.ylabel('film quantity')
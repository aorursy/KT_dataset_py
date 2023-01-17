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
# import packages

import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import matplotlib

plt.style.use('ggplot')

from matplotlib.pyplot import figure



%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (12,8)



pd.options.mode.chained_assignment = None







# read the data

df = pd.read_csv("../input/sberbank_train.csv")



# shape and data types of the data

print(df.shape)

print(df.dtypes)



# select numeric columns

numeric_cols = df.select_dtypes(include=[np.number]).columns.values

#print(numeric_cols)



# select non numeric columns

non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.values

#print(non_numeric_cols)
cols = df.columns[:30] # first 30 columns

colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.

sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
# if it's a larger dataset and the visualization takes too long can do this.

# % of missing.

for col in df.columns:

    pct_missing = np.mean(df[col].isnull())

    print('{} - {}%'.format(col, round(pct_missing*100)))
# first create missing indicator for features with missing data

for col in df.columns:

    missing = df[col].isnull()

    num_missing = np.sum(missing)

    

    if num_missing > 0:  

        #print('created missing indicator for: {}'.format(col))

        df['{}_ismissing'.format(col)] = missing





# then based on the indicator, plot the histogram of missing values

ismissing_cols = [col for col in df.columns if 'ismissing' in col]

df['num_missing'] = df[ismissing_cols].sum(axis=1)



df['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')
# drop rows with a lot of missing values.

ind_missing = df[df['num_missing'] > 35].index

df_less_missing_rows = df.drop(ind_missing, axis=0)

df_less_missing_rows['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')
# hospital_beds_raion has a lot of missing.

# If we want to drop.

cols_to_drop = ['hospital_beds_raion']

df_less_hos_beds_raion = df.drop(cols_to_drop, axis=1)
# replace missing values with the median.

med = df['life_sq'].median()

print(med)

df['life_sq'] = df['life_sq'].fillna(med)



# impute the missing values and create the missing value indicator variables for each numeric column.

numeric_cols = df.select_dtypes(include=[np.number]).columns.values



for col in numeric_cols:

    med = df[col].median()

    df[col] = df[col].fillna(med)



non_numeric_cols  = df.select_dtypes(exclude=[np.number]).columns.values



# impute the missing values and create the missing value indicator variables for each non-numeric column.

for col in non_numeric_cols:

    top = df[col].describe()['top'] # impute with the most frequent value.

    df[col] = df[col].fillna(top)

# categorical

df['sub_area'] = df['sub_area'].fillna('_MISSING_')



# numeric

df['life_sq'] = df['life_sq'].fillna(-999)
# histogram of life_sq.

df['life_sq'].hist(bins=100)
# box plot.

df.boxplot(column=['life_sq'])
df['life_sq'].describe()
# bar chart -  distribution of a categorical variable

df['ecology'].value_counts().plot.bar()
num_rows = len(df.index)

low_information_cols = [] #



for col in df.columns:

    cnts = df[col].value_counts(dropna=False)

    top_pct = (cnts/num_rows).iloc[0]

    

    if top_pct > 0.95:

        low_information_cols.append(col)

        print('{0}: {1:.5f}%'.format(col, top_pct*100))

        print(cnts)

        print()
# we know that column 'id' is unique, but what if we drop it?

df_dedupped = df.drop('id', axis=1).drop_duplicates()



# there were duplicate rows

print(df.shape)

print(df_dedupped.shape)
key = ['timestamp', 'full_sq', 'life_sq', 'floor', 'build_year', 'num_room', 'price_doc']



df.fillna(-999).groupby(key)['id'].count().sort_values(ascending=False).head(20)
# drop duplicates based on an subset of variables.



key = ['timestamp', 'full_sq', 'life_sq', 'floor', 'build_year', 'num_room', 'price_doc']

df_dedupped2 = df.drop_duplicates(subset=key)



print(df.shape)

print(df_dedupped2.shape)
print(df['sub_area'].value_counts(dropna=False))



# make everything lower case.

df['sub_area_lower'] = df['sub_area'].str.lower()

df['sub_area_lower'].value_counts(dropna=False)

df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')

df['year'] = df['timestamp_dt'].dt.year

df['month'] = df['timestamp_dt'].dt.month

df['weekday'] = df['timestamp_dt'].dt.weekday



print(df['year'].value_counts(dropna=False))

print()

print(df['month'].value_counts(dropna=False))
from nltk.metrics import edit_distance



df_city_ex = pd.DataFrame(data={'city': ['torontoo', 'toronto', 'tronto', 'vancouver', 'vancover', 'vancouvr', 'montreal', 'calgary']})





df_city_ex['city_distance_toronto'] = df_city_ex['city'].map(lambda x: edit_distance(x, 'toronto'))

df_city_ex['city_distance_vancouver'] = df_city_ex['city'].map(lambda x: edit_distance(x, 'vancouver'))

df_city_ex
msk = df_city_ex['city_distance_toronto'] <= 2

df_city_ex.loc[msk, 'city'] = 'toronto'



msk = df_city_ex['city_distance_vancouver'] <= 2

df_city_ex.loc[msk, 'city'] = 'vancouver'



df_city_ex
# no address column in the housing dataset. So create one to show the code.

df_add_ex = pd.DataFrame(['123 MAIN St Apartment 15', '123 Main Street Apt 12   ', '543 FirSt Av', '  876 FIRst Ave.'], columns=['address'])

print(df_add_ex)



df_add_ex['address_std'] = df_add_ex['address'].str.lower()

df_add_ex['address_std'] = df_add_ex['address_std'].str.strip() # remove leading and trailing whitespace.

df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\.', '') # remove period.

df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bstreet\\b', 'st') # replace street with st.

df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bapartment\\b', 'apt') # replace apartment with apt.

df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bav\\b', 'ave') # replace apartment with apt.



print(df_add_ex)

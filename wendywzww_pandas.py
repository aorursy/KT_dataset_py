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
dict = {'color' : ['black', 'white', 'black', 'white', 'black',

                      'white', 'black', 'white', 'black', 'white'],

            'size' : ['S','M','L','M','L','S','S','XL','XL','M'],

        'date':pd.date_range('1/1/2019',periods=10, freq='W' ),

            'feature_1': np.random.randn(10),

            'feature_2': np.random.normal(0.5, 2, 10)}

array=[['A','B','B','B','C','A','B','A','C','C'],['JP','CN','US','US','US','CN','CN','CA','JP','CA']]

                                                 

index = pd.MultiIndex.from_arrays(array, names=['class', 'country'])

df = pd.DataFrame(dict,index=index)

df
group_1 = df.groupby('size')

print(list(group_1))
group_1.sum().add_prefix('sum_')
group_1.get_group('M')
group_2 = df.groupby(['size', 'color'])

print(list(group_2))
print(group_1.size())

print(group_2.size())
def get_letter_type(letter):

    if 'feature' in letter:

        return 'feature'

    else:

        return 'other'

print(list(df.groupby(get_letter_type, axis=1)))
print(list(df.groupby(level='class')))
group_3=df.groupby(['country','color'])

for name, group in group_3:

    print(name)

    print(group)
group_3.agg({'feature_1' : np.min,'feature_2' : np.mean})
data_range = lambda x: x.max() - x.min()

df.groupby('size').transform(data_range)
df.iloc[1, 3:5] = np.NaN

f = lambda x: x.fillna(x.mean())

df_trans = group_1.transform(f)

df_trans
df.groupby('color').rolling(3).feature_1.mean()
group_2.expanding().sum()
df.groupby('class').filter(lambda x: len(x) > 3)
df.groupby('class')['feature_1'].apply(lambda x: x.describe())
def f(group):

    return pd.DataFrame({'original' : group,'demeaned' : group - group.mean()})

df.groupby('class')['feature_1'].apply(f)
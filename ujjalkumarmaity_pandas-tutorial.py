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
import pandas as pd



df = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

df.head()
df.head(2)
df['points']
print(df.points[1])

df.points
df.loc[1]
df.iloc[1]
df.loc[1:6,'points']
df.loc[1:3,['country' ,'description' ,'designation' ,'points']]
df.iloc[1:4,0:4]
df.iloc[[1,2,5,7],0]
df.iloc[-3:]
df.set_index('country').head(2)

df[df.country=='US'].head(2)

#or

#df.loc[df.country=='US']
df[(df.country=='US') & (df.points>=87)].head(2)
df[(df.country=='Italy') | (df.points>=90)].head(2)
df.describe()
df.points.describe()
df.country.unique()
df.taster_twitter_handle.value_counts()
df.points.mean()
def fun(data):

    if data.points>90:

        data.points=0

    else:

        data.points=1

    return data

df.apply(fun,axis='columns').head(4)
df.groupby('points').points.count()
df.groupby('points').price.max()
df.dtypes
df.index.dtype
df.points.astype('float')
column=df.columns

for i in column:

    print(i,df[i].isnull().sum())
df.price.fillna(df.price.mean(),inplace=True)
df.region_1.fillna('Unknown',).value_counts()
renamed = df.rename(columns={'region_1':'region','region_2':'locale'})

renamed.head(1)
reindexed = df.rename_axis('wines',axis='columns')

reindexed.head(1)
df.rename(index={0:'first',1:'second'}).head(1)
#total=pd.concat([dataset1,datset2])
df.groupby('points').points.count()
import pandas as pd

winemag_data_2 = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")

winemag_data = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
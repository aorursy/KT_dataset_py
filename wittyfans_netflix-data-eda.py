# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import squarify as sq







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/netflix-shows/netflix_titles_nov_2019.csv')
df.info()
df.describe()
df.head()
dup_title_df = df[df.duplicated('title',keep=False)]

dup_title_df = dup_title_df[['title','director','show_id']]

dup_title_df.groupby(['title','director']).show_id.count().to_frame()
director_df = df.groupby('country').director.value_counts().to_frame()

director_df.columns = ['num']

director_df.sort_values(by='num',ascending=False).head(10)
df.type.unique()
# all tv casts

tv_casts = df[df.type=='TV Show'].cast.str.split(',').tolist()

tv_casts = [c for c in tv_casts if type(c) != float]

tv_casts = np.concatenate(tv_casts)

tv_casts = pd.Series(tv_casts,name='name')



# all movie casts

mv_casts = df[df.type=='Movie'].cast.str.split(',').tolist()

mv_casts = [c for c in mv_casts if type(c) != float]

mv_casts = np.concatenate(mv_casts)

mv_casts = pd.Series(mv_casts,name='name')



# all casts

casts = mv_casts.append(tv_casts)

casts.value_counts().to_frame().head(10)
val_counts = {}

val_counts['index'] = ['num '+str(i) for i in range(10)]

for col in df.columns:

    val = df[col].value_counts().head(10).index

    if (len(val)==10) and (col not in ['cast','listed_in','description']):

        val_counts[col]=val



val_count_df = pd.DataFrame(val_counts).set_index('index')

val_count_df
type_val_count = df.type.value_counts(normalize=True)

plt.figure(figsize=(7.5,7.5))

type_val_count.plot(kind='pie',autopct='%1.1f%%')
tv_df = df[(df.type=='TV Show') & (~df.date_added.isnull())]

tv_df.date_added = pd.to_datetime(tv_df.date_added)

tv_df = tv_df.set_index('date_added')

plt.figure(figsize=(20,8))

tv_added_df = tv_df.resample('1M').show_id.count()



mv_df = df[(df.type=='Movie') & (~df.date_added.isnull())]

mv_df.date_added = pd.to_datetime(mv_df.date_added)

mv_df = mv_df.set_index('date_added')

mv_add_df = mv_df.resample('1M').show_id.count()



plt.plot(tv_added_df.index,tv_added_df.values,label='TV added')

plt.plot(mv_add_df.index,mv_add_df.values,label='Movie added')

plt.legend(loc='lower right')
country_share_pct = df.country.value_counts(normalize=True).head(10)

plt.figure(figsize=(30,12))

sns.barplot(x=country_share_pct.index,y=country_share_pct.values)

plt.xlabel('Country')

plt.ylabel('Share')
top_countrys = country_share_pct.index

plt.figure(figsize=(20,35))

for index,country in enumerate(top_countrys):

    country_df = df[df.country==country]

    country_df_casts = country_df.cast.str.split(',').tolist()

    country_df_casts = [c for c in country_df_casts if type(c) != float]

    country_df_casts = np.concatenate(country_df_casts)

    country_df_casts = pd.Series(country_df_casts,name='name')

    country_polular_casts = country_df_casts.value_counts().head(10)

    plt.subplot(6,2,index+1)

    plt.barh(country_polular_casts.index,country_polular_casts.values)

    plt.title('Most popular cast in {}'.format(country))

    plt.xlabel('Invoved Show')

    plt.ylabel('Actors Name')
listed = [x.split(',') for x in df.listed_in.values if len(x)>1]

list_concated = np.concatenate(listed)

list_concated = pd.Series(list_concated)

list_val_count = list_concated.value_counts().head(50)

plt.figure(figsize=(30,15))

plt.axis('off')

sq.plot(sizes=list_val_count,label=list_val_count.index,alpha=0.7)
top_countrys = country_share_pct.head(6).index

plt.figure(figsize=(30,15))

for index,country in enumerate(top_countrys):

    country_df = df[df.country==country]

    country_polular_rating = country_df.rating.value_counts()

    plt.subplot(2,3,index+1)

#     country_polular_rating.plot(kind='pie',autopct='%1.1f%%')

    plt.bar(country_polular_rating.index,country_polular_rating.values)

    plt.title('Most popular Rating shows in {}'.format(country))

    plt.xlabel('')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
## connection

conn = sqlite3.connect('../input/database.sqlite')

tag = pd.read_sql_query('Select * From tags', conn)

torrents = pd.read_sql_query('Select * From torrents',conn)
torrents.head()
ax = sns.countplot(torrents['releaseType'])

plt.title('Release Type Distribution')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
table_count = pd.pivot_table(data=torrents,

                             index=['groupYear'],

                             columns=['releaseType'],

                             values=['totalSnatched'],

                             aggfunc='count')



fig,ax = plt.subplots(figsize=(8,6))

plt.title('GroupYear vs ReleaseType')

sns.heatmap(table_count['totalSnatched'],vmin=0,linewidth=.5,annot=True,fmt='2.0f')
sns.set(style="white")

fig,ax = plt.subplots(figsize=(8,6))

g = sns.countplot(data = torrents,

               x = 'groupYear',

               palette="BuPu",ax=ax)

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

plt.title('Year Distribution')
artist_counts = torrents['artist'].value_counts().sort_values(ascending=False)[:20]

artist_name = artist_counts.index



table_count = pd.pivot_table(data = torrents[torrents['artist'].isin(artist_name)],

                             index=['releaseType'],

                             columns=['artist'],

                             values=['totalSnatched'],

                             aggfunc='count')



sns.heatmap(table_count['totalSnatched'],vmin=0,annot=True,fmt='2.0f')

plt.title('Artist vs ReleaseType')
artist_counts = torrents['artist'].value_counts().sort_values(ascending=False)[:20]

artist_name = artist_counts.index



table_count = pd.pivot_table(data = torrents[torrents['artist'].isin(artist_name)],

                             index=['groupYear'],

                             columns=['artist'],

                             values=['totalSnatched'],

                             aggfunc='count')



sns.heatmap(table_count['totalSnatched'],vmin=0,annot=True,fmt='2.0f')

plt.title('Artist vs groupYear')
tag.head()
all_data = pd.merge(torrents,tag,on='id')

all_data.head()
tag_counts = all_data['tag'].value_counts().sort_values(ascending=False)[:20]

tag_index = tag_counts.index



table_count = pd.pivot_table(all_data[all_data['tag'].isin(tag_index)],

                             index=['groupYear'],

                             columns=['tag'],

                             values=['totalSnatched'],

                             aggfunc='count')

sns.heatmap(table_count['totalSnatched'],vmin=0,annot=False)

plt.title('Tag vs GroupYear')
all_data['cut_Year'] = pd.cut(all_data['groupYear'],[1978,1990,2001,2012,2017],

                              labels=['1979-1989','1990-2000','2001-2011','2012+'])
sns.set(style="white")

fig,ax = plt.subplots(figsize=(8,6))

g = sns.countplot(data = all_data,

               x = 'cut_Year',

               palette="BuPu",ax=ax)

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

plt.title('Year Distribution')
fig,ax=plt.subplots(figsize=(8,6))

ax = sns.countplot(data=all_data, x ='releaseType',hue='cut_Year',ax=ax)

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

plt.legend(loc='best',bbox_to_anchor=(1.05, 1))
year_index = all_data['groupYear'].value_counts().sort_values(ascending=False)[:20].index



table_count = pd.pivot_table(data=all_data[all_data['groupYear'].isin(year_index)],

                             index=['groupYear'],

                             columns=['releaseType'],

                             values=['totalSnatched'],

                             aggfunc='sum')





sns.heatmap(table_count['totalSnatched'],vmin=0,linewidth=.5,annot=False)

plt.title('Group Year vs ReleaseType and TotalSnatched')
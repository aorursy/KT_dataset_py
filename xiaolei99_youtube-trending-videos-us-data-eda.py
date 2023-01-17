# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import matplotlib.pyplot as plt

from matplotlib import cm

from datetime import datetime

import glob

import seaborn as sns

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
files = [i for i in glob.glob('../input/youtube-new/*.{}'.format('csv'))]

sorted(files)
dfs = []



for csv in files:

    try:

        df = pd.read_csv(csv, index_col='video_id', engine='python')

        # engine = 'python' will allow the parser to recognize languages like korean and japanese

    except: 

        df = pd.read_csv(csv, index_col='video_id', encoding='UTF-8')

        # encoding='UTF-8' is the default, recognize languages like English

    df['country'] = csv[21:23]

    dfs.append(df)

    

all_df = pd.concat(dfs)

all_df.head(2)
all_df.shape
dateparse1 = lambda x: pd.datetime.strptime(x, '%y.%d.%m')

all_df['trending_date'] = all_df.trending_date.apply(dateparse1)

dateparse2 = lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')

all_df['publish_time'] = all_df.publish_time.apply(dateparse2)

all_df.insert(4, 'publish_date', all_df.publish_time.dt.date)

all_df['publish_time'] = all_df.publish_time.dt.time

all_df.head()
full_df = all_df.reset_index().sort_values('trending_date').set_index('video_id')

df = all_df.reset_index().sort_values('trending_date').drop_duplicates('video_id', keep='last').set_index('video_id')

df.head()
import json

with open('/kaggle/input/youtube-new/US_category_id.json') as json_file:

    data = json.load(json_file)



data['items'][3]['id']

for i in range(len(data['items'])):

    print(data['items'][i]['snippet']['title'])



id = []



for i in range(32):

    id.append(data['items'][i]['id'])

    

title = []

for i in range(32):

    title.append(data['items'][i]['snippet']['title'])



d = {'id': id, 'title': title}

df2 = pd.DataFrame(data=d)



df2.id = df2.id.astype(int)

df2.dtypes



df3 = pd.merge(df,df2, left_on='category_id', right_on='id')



df3.head()

df3.insert(4, 'category', df3.title_y)

df3.drop(columns='title_y')

df = df3

df = df.set_index('trending_date').drop(columns=['title_y'])

df = df.drop(columns=['id'])
df.category.unique()
df2 = df.groupby(by='category').count().sort_values(by='title_x', ascending=False)

sns.set_style("whitegrid")

ax = sns.barplot(y=df2.index,x=df2.title_x, data= df2,orient='h')

plt.xlabel("Number of Videos")## From United Kingdom users : 

plt.ylabel("Categories")

plt.title("Categories of trending videos globally")
df.sort_values(by='views', ascending=False).head(100).groupby(by='category').count()[['title_x']]
df_val = df.sort_values(by='views', ascending=False).head(100).groupby(by='category').count().title_x

df_index = df.sort_values(by='views', ascending=False).head(100).groupby(by='category').count().index
plt.pie(df_val, labels=df_index, radius = 2.5, autopct = '%0.2f%%')

plt.show()
df.country.unique()
df_US = df[df.country == 'US'].groupby(by='category').count().sort_values(by='title_x', ascending=False)[['title_x']].head(3)

df_MX = df[df.country == 'MX'].groupby(by='category').count().sort_values(by='title_x', ascending=False)[['title_x']].head(3)

df_GB = df[df.country == 'GB'].groupby(by='category').count().sort_values(by='title_x', ascending=False)[['title_x']].head(3)

df_KR = df[df.country == 'KR'].groupby(by='category').count().sort_values(by='title_x', ascending=False)[['title_x']].head(3)

df_FR = df[df.country == 'FR'].groupby(by='category').count().sort_values(by='title_x', ascending=False)[['title_x']].head(3)

df_DE = df[df.country == 'DE'].groupby(by='category').count().sort_values(by='title_x', ascending=False)[['title_x']].head(3)

df_RU = df[df.country == 'RU'].groupby(by='category').count().sort_values(by='title_x', ascending=False)[['title_x']].head(3)

df_IN = df[df.country == 'IN'].groupby(by='category').count().sort_values(by='title_x', ascending=False)[['title_x']].head(3)

df_CA = df[df.country == 'CA'].groupby(by='category').count().sort_values(by='title_x', ascending=False)[['title_x']].head(3)

df_JP = df[df.country == 'JP'].groupby(by='category').count().sort_values(by='title_x', ascending=False)[['title_x']].head(3)

fig, ax =plt.subplots(1,2, figsize=(20,8))

sns.barplot(y=df_US.index,x=df_US.title_x, data= df_US,orient='h', ax=ax[0]).set_title("Top Categories in US")

sns.barplot(y=df_MX.index,x=df_MX.title_x, data= df_MX,orient='h', ax=ax[1]).set_title("Top Categories in MX")

ax[0].set(xlabel="Videos")

ax[1].set(xlabel="Videos")







fig, ax =plt.subplots(1,2, figsize=(20,8))

sns.barplot(y=df_GB.index,x=df_GB.title_x, data= df_GB,orient='h', ax=ax[0]).set_title("Top Categories in GB")

sns.barplot(y=df_KR.index,x=df_KR.title_x, data= df_KR,orient='h', ax=ax[1]).set_title("Top Categories in KR")

ax[0].set(xlabel="Videos")

ax[1].set(xlabel="Videos")



fig, ax =plt.subplots(1,2, figsize=(20,8))

sns.barplot(y=df_DE.index,x=df_DE.title_x, data= df_DE,orient='h', ax=ax[0]).set_title("Top Categories in DE")

sns.barplot(y=df_RU.index,x=df_RU.title_x, data= df_RU,orient='h', ax=ax[1]).set_title("Top Categories in RU")

ax[0].set(xlabel="Videos")

ax[1].set(xlabel="Videos")



fig, ax =plt.subplots(1,2, figsize=(20,8))

sns.barplot(y=df_IN.index,x=df_IN.title_x, data= df_IN,orient='h', ax=ax[0]).set_title("Top Categories in IN")

sns.barplot(y=df_CA.index,x=df_CA.title_x, data= df_CA,orient='h', ax=ax[1]).set_title("Top Categories in CA")

plt.xlabel('Videos')

ax[0].set(xlabel="Videos")

ax[1].set(xlabel="Videos")

sns.barplot(y=df_JP.index,x=df_JP.title_x, data= df_JP,orient='h').set_title("Top Categories in JP")

plt.xlabel('Videos')

ax[0].set(xlabel="Videos")
all_df = all_df.dropna(how='any',inplace=False, axis = 0)
all_df.shape
df2_US = all_df[all_df.country == "US"].groupby(by='video_id').count().sort_values(by='title', ascending=False)[['title']]

df2_US = all_df[all_df.country == "US"].groupby(by='video_id').count().sort_values(by='title', ascending=False)[['title']]

df2_US.title.value_counts().sort_index()

plt.plot(df2_US.title.value_counts().index, df2_US.title.value_counts())

plt.xlabel('Number of Days')

plt.ylabel('Number of Videos')

plt.title('Number of Days Videos Trend in US')
df2_RU = all_df[all_df.country == "RU"].groupby(by='video_id').count().sort_values(by='title', ascending=False)[['title']]

df2_RU.title.value_counts().sort_index()

plt.plot(df2_RU.title.value_counts().index, df2_RU.title.value_counts())

axes = plt.gca()

axes.set_xlim([0,10])

plt.xlabel('Number of Days')

plt.ylabel('Number of Videos')

plt.title('Number of Days Videos Trend in RU')

df2_MX = all_df[all_df.country == "MX"].groupby(by='video_id').count().sort_values(by='title', ascending=False)[['title']]

df2_MX.title.value_counts().sort_index()

plt.plot(df2_MX.title.value_counts().index, df2_MX.title.value_counts())

axes = plt.gca()

axes.set_xlim([0,10])

plt.xlabel('Number of Days')

plt.ylabel('Number of Videos')

plt.title('Number of Days Videos Trend in MX')
df2_GB = all_df[all_df.country == "GB"].groupby(by='video_id').count().sort_values(by='title', ascending=False)[['title']]

df2_GB.title.value_counts().sort_index()

plt.plot(df2_GB.title.value_counts().index, df2_GB.title.value_counts())

axes = plt.gca()

axes.set_xlim([0,39])

plt.xlabel('Number of Days')

plt.ylabel('Number of Videos')

plt.title('Number of Days Videos Trend in GB')
df2_FR = all_df[all_df.country == "FR"].groupby(by='video_id').count().sort_values(by='title', ascending=False)[['title']]

df2_FR.title.value_counts().sort_index()

plt.plot(df2_FR.title.value_counts().index, df2_FR.title.value_counts())

axes = plt.gca()

axes.set_xlim([0,15])

plt.xlabel('Number of Days')

plt.ylabel('Number of Videos')

plt.title('Number of Days Videos Trend in GB')
df2_DE = all_df[all_df.country == "DE"].groupby(by='video_id').count().sort_values(by='title', ascending=False)[['title']]

df2_DE.title.value_counts().sort_index()

plt.plot(df2_DE.title.value_counts().index, df2_DE.title.value_counts())

axes = plt.gca()

axes.set_xlim([0,15])

plt.xlabel('Number of Days')

plt.ylabel('Number of Videos')

plt.title('Number of Days Videos Trend in DE')
df2_CA = all_df[all_df.country == "CA"].groupby(by='video_id').count().sort_values(by='title', ascending=False)[['title']]

df2_CA.title.value_counts().sort_index()

plt.plot(df2_CA.title.value_counts().index, df2_CA.title.value_counts())

axes = plt.gca()

axes.set_xlim([0,15])

plt.xlabel('Number of Days')

plt.ylabel('Number of Videos')

plt.title('Number of Days Videos Trend in CA')
df2_IN = all_df[all_df.country == "IN"].groupby(by='video_id').count().sort_values(by='title', ascending=False)[['title']]

df2_IN.title.value_counts().sort_index()

plt.plot(df2_IN.title.value_counts().index, df2_IN.title.value_counts())

axes = plt.gca()

axes.set_xlim([0,15])

plt.xlabel('Number of Days')

plt.ylabel('Number of Videos')

plt.title('Number of Days Videos Trend in IN')
df2_KR = all_df[all_df.country == "KR"].groupby(by='video_id').count().sort_values(by='title', ascending=False)[['title']]

df2_KR.title.value_counts().sort_index()

plt.plot(df2_KR.title.value_counts().index, df2_KR.title.value_counts())

axes = plt.gca()

axes.set_xlim([0,15])

plt.xlabel('Number of Days')

plt.ylabel('Number of Videos')

plt.title('Number of Days Videos Trend in KR')
df2_JP = all_df[all_df.country == "JP"].groupby(by='video_id').count().sort_values(by='title', ascending=False)[['title']]

df2_JP.title.value_counts().sort_index()

plt.plot(df2_JP.title.value_counts().index, df2_JP.title.value_counts())

axes = plt.gca()

axes.set_xlim([0,15])

plt.xlabel('Number of Days')

plt.ylabel('Number of Videos')

plt.title('Number of Days Videos Trend in JP')
df.groupby(by='country').count().title_x.sort_values(ascending=False)
df3 = pd.merge(df,df2, left_on='category_id', right_on='id')
df3.set_index('trending_date', inplace=True)
df3['2017-11-14'][['title_x', 'views', 'title_y']].head()
import numpy as np

import datetime as dt
df3[df3.title_y == "Music"].resample('M').count().title_y

xpos = np.arange(len(df3[df3.title_y == "Music"].resample('M').count().title_y))

plt.bar(xpos, df3[df3.title_y == "Music"].resample('M').count().title_y, width = 0.15, label = 'Music')

plt.bar(xpos + 0.15, df3[df3.title_y == "Entertainment"].resample('M').count().title_y, width = 0.15, label = 'Entertainment')

plt.bar(xpos + 0.3, df3[df3.title_y == "Howto & Style"].resample('M').count().title_y, width = 0.15, label = 'Howto & Style')

plt.bar(xpos - 0.15, df3[df3.title_y == "People & Blogs"].resample('M').count().title_y, width = 0.15, label = 'People & Blogs')

plt.legend()

dates = df3[df3.title_y == "Music"].resample('M').count().title_y.index.month

plt.xticks(xpos, dates)

dates

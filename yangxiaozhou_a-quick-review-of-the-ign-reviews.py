import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

%matplotlib inline
sns.set_palette('Set3', 10)

sns.palplot(sns.color_palette())

sns.set_context('talk')
raw_data = pd.read_csv('../input/ign.csv')
raw_data.head()
release_date = raw_data.apply(lambda x: pd.datetime.strptime("{0} {1} {2} 00:00:00".format(

            x['release_year'],x['release_month'], x['release_day']), "%Y %m %d %H:%M:%S"),axis=1)

raw_data['release_date'] = release_date
raw_data[raw_data.release_year == 1970]
data = raw_data[raw_data.release_year > 1970]

len(data)
data.score_phrase.unique()
data.groupby('score_phrase')['score'].mean().sort_values()
data.platform.unique()
plt.figure(figsize=(15,8))

data.groupby(['release_day']).size().plot(c='r')

plt.xticks(range(1,32,3))

plt.tight_layout()
f, ax = plt.subplots(2,1,figsize=(15,10),sharex=True)

data.release_date.dt.weekday.plot.kde(ax=ax[0],c='g')

data.groupby(data.release_date.dt.weekday).size().plot(ax=ax[1],c='r')

plt.xlim(0.,6.)

plt.xticks(range(7),['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.tight_layout()
plt.figure(figsize=(17,8))

plt.xticks(range(1,13),['January','February','March','April','May','June',

            'July','August','September','October','November','December'])

data.groupby(['release_month']).size().plot(c='r')
plt.figure(figsize=(17,8))

data.groupby(['release_year']).size().plot(kind='bar')
table = data.groupby('release_date').size()

f,ax = plt.subplots(2,1,figsize=(17,10))

#table.rolling(window=30).mean().plot(c='orange',ax=ax[1])

table.plot(ax=ax[0],c='red')

ax[0].set_xlabel('')

table.resample('M').mean().plot(c='orange',ax=ax[1])
data.platform.value_counts()[:10].plot.pie(figsize=(10,10))
f, ax = plt.subplots(2,2, figsize=(17,17))

last_games = data[data.release_year == 2014]

last_popular = last_games.platform.value_counts()[last_games.platform.value_counts() > 5]

last_popular.plot.pie(ax=ax[0,0])

ax[0,0].set_title('2014')

ax[0,0].set_ylabel('')

last_games = data[data.release_year == 2015]

last_popular = last_games.platform.value_counts()[last_games.platform.value_counts() > 5]

last_popular.plot.pie(ax=ax[0,1])

ax[0,1].set_title('2015')

ax[0,1].set_ylabel('')

last_games = data[data.release_year == 2016]

last_popular = last_games.platform.value_counts()[last_games.platform.value_counts() > 5]

last_popular.plot.pie(ax=ax[1,0])

ax[1,0].set_title('2016')

ax[1,0].set_ylabel('')

old_games = data[data.release_year <= 2000]

old_popular = old_games.platform.value_counts()[old_games.platform.value_counts() > 5]

old_popular.plot.pie(ax=ax[1,1])

ax[1,1].set_title('2000 and older')

ax[1,1].set_ylabel('')
years = tuple(range(1996,2017))

s = data.groupby([data.release_year,data.platform]).title.count()

top_years_platform = pd.DataFrame([[i,s[i].max(),s[i].argmax()] for i in years], 

                                 columns=['release_year','count_games','platform'])



sc = data.groupby([data.release_year,data.platform]).score

s = sc.median()[sc.count() > 20]

top_scores_platform = pd.DataFrame([[i,s[i].max(),s[i].argmax()] for i in years], 

                                 columns=['release_year','score_game','platform'])
f, axes = plt.subplots(1,2,figsize=(18,20))



ax = top_years_platform.count_games.plot(kind='barh',color='orange',ax=axes[0])

ax.set_yticklabels(years) 

ax.set_xlabel('Count of releases')

rects = ax.patches

for i, v in enumerate(top_years_platform.platform): 

    ax.text(10, i-.1, v, fontweight='bold')



ax2 = top_scores_platform.score_game.plot(kind='barh',color='blue',ax=axes[1])

ax2.set_yticklabels(years) 

ax2.set_xlabel('Average score')

rects = ax2.patches

for i, v in enumerate(top_scores_platform.platform): 

    ax2.text(0.3, i-.1, v, fontweight='bold', color='white')
data_pc = data[data.platform == 'PC']

data_ps = data[data.platform == 'PlayStation']

data_ps2 = data[data.platform == 'PlayStation 2']

data_ps3 = data[data.platform == 'PlayStation 3']

data_ps4 = data[data.platform == 'PlayStation 4']

data_xbox = data[data.platform == 'Xbox']

data_xbox360 = data[data.platform == 'Xbox 360']

data_xbox_one = data[data.platform == 'Xbox One']

df = pd.DataFrame({'PC' : data_pc.groupby('release_year').size(),

                   'PS' : data_ps.groupby('release_year').size(),

                   'PS2' : data_ps2.groupby('release_year').size(),

                   'PS3' : data_ps3.groupby('release_year').size(),

                   'PS4' : data_ps4.groupby('release_year').size(),

                   'Xbox' : data_xbox.groupby('release_year').size(),

                   'Xbox 360' : data_xbox360.groupby('release_year').size(),

                   'Xbox One' : data_xbox_one.groupby('release_year').size()

                  })
f,ax = plt.subplots(1,1,figsize=(15,20))

df.plot(kind='barh',stacked=True,ax=ax)
data_pc = data[data.platform == 'PC']

plt.figure(figsize=(15,8))

data_pc.groupby('release_year').platform.size().plot(kind='bar',color='green')
plt.figure(figsize=(15,8))

plt.xlim(1995,2017)

plt.ylim(1.8,10)

sns.kdeplot(data.release_year, data.score, n_levels=20, cmap="Reds", shade=True, shade_lowest=False)
plt.figure(figsize=(15,8))

plt.ylim(1.5,10.5)

plt.xticks(range(1,13),['January','February','March','April','May','June',

            'July','August','September','October','November','December'])

sns.kdeplot(data.release_month, data.score, n_levels=20, cmap="Blues", shade=True, shade_lowest=False)
plt.figure(figsize=(15,8))

plt.ylim(1.5,10.5)

sns.kdeplot(data.release_day, data.score, n_levels=20, cmap="Greens", shade=True, shade_lowest=False)
plt.figure(figsize=(17,8))

#sns.kdeplot(data.score, shade=True, c='g', label='Density')

plt.xticks(np.linspace(0,10,21))

plt.xlim(0,10)

data.score.plot.kde(c='g', label='Density')

plt.legend()
plt.figure(figsize=(17,10))

plt.xticks(np.linspace(0,10,21))

plt.xlim(0,10)

data.score.plot.kde(label='All platform')

data[data.platform == 'PC'].score.plot.kde(label='PC')

#data[data.platform == 'PlayStation'].score.plot.kde(label='PlayStation')

#data[data.platform == 'PlayStation 2'].score.plot.kde(label='PlayStation 2')

data[data.platform == 'PlayStation 3'].score.plot.kde(label='PlayStation 3')

data[data.platform == 'PlayStation 4'].score.plot.kde(label='PlayStation 4')

plt.legend(loc='upper left')
plt.figure(figsize=(17,10))

plt.xticks(np.linspace(0,10,21))

plt.xlim(0,10)

data.score.plot.kde(label='All platform')

data[data.platform == 'PC'].score.plot.kde(label='PC')

#data[data.platform == 'Xbox'].score.plot.kde(label='Xbox')

data[data.platform == 'Xbox 360'].score.plot.kde(label='Xbox 360')

data[data.platform == 'Xbox One'].score.plot.kde(label='Xbox One')

plt.legend(loc='upper left')
plt.figure(figsize=(17,10))

plt.xticks(np.linspace(0,10,21))

plt.xlim(0,10)

data.score.plot.kde(label='All platform')

data[data.platform == 'Android'].score.plot.kde(label='Android')

data[data.platform == 'iPhone'].score.plot.kde(label='iPhone')

data[data.platform == 'iPad'].score.plot.kde(label='iPad')

plt.legend(loc='upper left')
plt.figure(figsize=(17,10))

plt.xticks(np.linspace(0,10,21))

plt.xlim(0,10)

data.score.plot.kde(label='All platform',c='black')

data[data.platform == 'PC'].score.plot.kde(label='PC')

data[data.platform == 'PlayStation 4'].score.plot.kde(label='PlayStation 4')

data[data.platform == 'Xbox One'].score.plot.kde(label='Xbox One')

data[data.platform == 'iPad'].score.plot.kde(label='iPad')

plt.legend(loc='upper left')
genres = data.groupby('genre')['genre']

genres_count=genres.count()

large_genres=genres_count[genres_count>=150]

large_genres.sort_values(ascending=False,inplace=True)

large_genres
data_genre = data[data.genre.isin(large_genres.keys())]

table_score = pd.pivot_table(data_genre,values=['score'],index=['release_year'],columns=['genre'],aggfunc='mean',margins=False)

table_count = pd.pivot_table(data_genre,values=['score'],index=['release_year'],columns=['genre'],aggfunc='count',margins=False)

table = table_score[table_count > 10]

plt.figure(figsize=(19,16))

sns.heatmap(table.score,linewidths=.5,annot=True,vmin=0,vmax=10,cmap='YlGnBu')

plt.title('Average scores of games (cell exists if a genre has at least 10 releases in year)')
plt.figure(figsize=(19,16))

sns.heatmap(table_count.score,linewidths=.5,annot=True,fmt='2.0f',vmin=0)

plt.title('Count of games')
import nltk
t = data.title.apply(nltk.word_tokenize).sum()
from collections import Counter

from string import punctuation



def content_text(text):

    stopwords = set(nltk.corpus.stopwords.words('english'))

    without_stp  = Counter()

    for word in text:

        word = word.lower()

        if len(word) < 3:

            continue

        if word not in stopwords:

            without_stp.update([word])

    return [(y,c) for y,c in without_stp.most_common(20)]



without_stop = content_text(t)

without_stop
from PIL import Image

import random

from wordcloud import WordCloud, STOPWORDS



text = ' '.join(t)

stopwords = set(STOPWORDS)



wordcloud = WordCloud(background_color='white', max_font_size=110, stopwords=stopwords, 

                      random_state=3, relative_scaling=.5).generate(text)

plt.figure(figsize=(15,18))

plt.imshow(wordcloud)

plt.axis('off')
master = data[data.score == 10][['title','platform','genre','release_year']]

master
f, ax = plt.subplots(2,1, figsize=(10,20))

master.groupby('genre').size().plot.pie(ax=ax[0],cmap='Set3')

master.groupby('platform').size().plot.pie(ax=ax[1],cmap='terrain')

ax[0].set_ylabel('')

ax[1].set_ylabel('')
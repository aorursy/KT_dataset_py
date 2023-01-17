# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

matplotlib.style.use('ggplot')

# Get current size

fig_size = plt.rcParams["figure.figsize"]

 

fig_size[0] = 4

fig_size[1] = 3

plt.rcParams["figure.figsize"] = fig_size

print ("Current size:", fig_size)
d = pd.read_csv("../input/ign.csv")

d = d[d.release_year > 1970]

print(d.head())
plot = pd.DataFrame(data=d[["release_year","genre"]])

s = plot['genre'].str.split(', ').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1) # to line up with df's index

s.name = 'genre' # needs a name to join

del plot['genre']

plot = plot.join(s)

plot.reset_index()

plot.genre.value_counts()[:9].plot.pie(cmap="viridis",figsize=(4,4))
plot = pd.DataFrame(data=d[["release_year","genre"]])

s = plot['genre'].str.split(', ').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1) # to line up with df's index

s.name = 'genre' # needs a name to join

del plot['genre']

plot = plot.join(s)

plot.reset_index()

plot = pd.DataFrame(data={'Year':plot.groupby(['release_year','genre']).size().index.get_level_values('release_year'),'Genre':plot.groupby(['release_year','genre']).size().index.get_level_values('genre'),'Count':plot.groupby(['release_year','genre']).size()})

plot = plot.pivot(index='Year', columns='Genre', values='Count')

plot = plot.replace(to_replace='NaN', value=0)

plot[['Action','Sports','Adventure','Shooter','Racing','RPG','Strategy','Puzzle','Platformer']].plot(kind="barh", stacked=True,cmap='viridis',legend=False)
release_date = d.apply(lambda x: pd.datetime.strptime("{0} {1} {2} 00:00:00".format(

            x['release_year'],x['release_month'], x['release_day']), "%Y %m %d %H:%M:%S"),axis=1)

d['release_date'] = release_date

d["release_date"].dt.weekday.plot.kde(cmap="viridis")

plt.xlim(0.,6.)

plt.xticks(range(7),['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rotation='vertical')
plot1 = pd.DataFrame(data=d["release_month"])

plot1 = pd.DataFrame(plot1.groupby(['release_month'], as_index = False).size().values)

plot1.index = plot1.index+1

plot1.columns = ['Overall game count']

plot1.plot(kind='bar', colormap='viridis')
d["score"].plot.kde(cmap='viridis', xlim=[0,10])
plot = pd.DataFrame(data=d[["release_year","score"]])

plot = pd.DataFrame(data={'Average':plot.groupby(['release_year']).mean()['score'],'Maximum':plot.groupby(['release_year']).max()['score'],'Minimum':plot.groupby(['release_year']).min()['score']})

plot = plot.replace(to_replace='NaN', value=0)

plot = plot[1:]



plot.plot(cmap='viridis')
plot = pd.DataFrame(data=d[["score","genre"]])

plot['score'] = plot['score'].round()

s = plot['genre'].str.split(', ').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1) # to line up with df's index

s.name = 'genre' # needs a name to join

del plot['genre']

plot = plot.join(s)

plot.reset_index()



plot = plot.replace(to_replace='NaN', value=0)

plot = plot.groupby(['genre']).mean()

plot = plot.sort_values('score',ascending=0)



plot[:10].plot(kind='bar',colormap='viridis',ylim=(0, 10))
plot = pd.DataFrame(data=d["score"])

plot = plot.replace(to_replace='NaN', value=0)



plot = plot.sort_values('score',ascending=0)

plot = plot.reset_index(drop=True)



plot.plot(kind='area', stacked=False, colormap='viridis', ylim=(0, 10))
plot = pd.DataFrame(data=d[["platform","genre"]])

s = plot['genre'].str.split(', ').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1) # to line up with df's index

s.name = 'genre' # needs a name to join

del plot['genre']

plot = plot.join(s)

plot.reset_index()



plot = pd.DataFrame(data={'Platform':plot.groupby(['platform','genre']).size().index.get_level_values('platform'),'Genre':plot.groupby(['platform','genre']).size().index.get_level_values('genre'),'Count':plot.groupby(['platform','genre']).size()})

plot = plot.reset_index(drop=True)

plot1 = plot[plot['Count']>20].pivot(index='Genre', columns='Platform', values='Count')

plot1 = plot1.replace(to_replace='NaN', value=0)

plot1.loc[:,:] = plot1.loc[:,:].div(plot1.sum(axis=1), axis=0)





plt.pcolor(plot1, cmap='viridis')

plt.yticks(np.arange(0.5, len(plot1.index), 1), plot1.index)

plt.xticks(np.arange(0.5, len(plot1.columns), 1), plot1.columns, rotation='vertical')

plt.show()
d.platform.value_counts()[:9].plot.pie(cmap="viridis",figsize=(4,4))
plot = pd.DataFrame({'PC' : d[d.platform == 'PC'].groupby('release_year').size(),

                   'PS' : d[d.platform == 'PlayStation'].groupby('release_year').size(),

                   'PS2' : d[d.platform == 'PlayStation 2'].groupby('release_year').size(),

                   'PS3' : d[d.platform == 'PlayStation 3'].groupby('release_year').size(),

                   'PS4' : d[d.platform == 'PlayStation 4'].groupby('release_year').size(),

                   'Xbox' : d[d.platform == 'Xbox'].groupby('release_year').size(),

                   'Xbox 360' : d[d.platform == 'Xbox 360'].groupby('release_year').size(),

                   'Xbox One' : d[d.platform == 'Xbox One'].groupby('release_year').size()

                  })

plot = plot.replace(to_replace='NaN', value=0)

plot.plot(kind="barh", stacked=1, cmap="viridis")
plot = pd.DataFrame(data=d[["editors_choice","score"]])

plot1 = pd.DataFrame(plot['score'][plot['editors_choice']=='Y'])

plot2 = pd.DataFrame(plot['score'][plot['editors_choice']=='N'])

plot1 = pd.DataFrame(data=plot1.groupby(['score']).size())

plot2 = pd.DataFrame(data=plot2.groupby(['score']).size())

plot1 = pd.concat([plot1, plot2], axis=1)

plot1.columns = ['Editors choice', 'Not editors choice']

plot1 = plot1.replace(to_replace='NaN', value=0)

plot1.loc[:,:] = plot1.loc[:,:].div(plot1.sum(axis=1), axis=0)

del plot1['Not editors choice']

plot1.plot(kind='area', cmap='viridis', stacked=True)
plot = pd.DataFrame(data=d[["editors_choice","release_year"]])

plot1 = pd.DataFrame(plot['release_year'][plot['editors_choice']=='Y'])

plot2 = pd.DataFrame(plot['release_year'][plot['editors_choice']=='N'])

plot1 = pd.DataFrame(data=plot1.groupby(['release_year']).size())

plot2 = pd.DataFrame(data=plot2.groupby(['release_year']).size())

plot1 = pd.concat([plot1, plot2], axis=1)

plot1.columns = ['Editors choice', 'Not editors choice']

plot1 = plot1.replace(to_replace='NaN', value=0)

plot1.loc[:,:] = plot1.loc[:,:].div(plot1.sum(axis=1), axis=0)

del plot1['Not editors choice']

plot1.plot(kind='bar', cmap='viridis', stacked=True)
plot = pd.DataFrame(data=d[["editors_choice","genre"]])

s = plot['genre'].str.split(', ').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1) # to line up with df's index

s.name = 'genre' # needs a name to join

del plot['genre']

plot = plot.join(s)

plot.reset_index()



plot1 = pd.DataFrame(plot['genre'][plot['editors_choice']=='Y'])

plot2 = pd.DataFrame(plot['genre'][plot['editors_choice']=='N'])

plot1 = pd.DataFrame(data=plot1.groupby(['genre']).size())

plot2 = pd.DataFrame(data=plot2.groupby(['genre']).size())

plot1 = pd.concat([plot1, plot2], axis=1)

plot1.columns = ['Editors choice', 'Not editors choice']

plot1 = plot1.replace(to_replace='NaN', value=0)

plot1.loc[:,:] = plot1.loc[:,:].div(plot1.sum(axis=1), axis=0)

plot1 = plot1.sort_values('Editors choice',ascending=0)

del plot1['Not editors choice']

plot1[:20].plot(kind='bar', cmap='viridis', stacked=True)
plot = pd.DataFrame(data=d[["score","genre"]])

s = plot['genre'].str.split(', ').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1) # to line up with df's index

s.name = 'genre' # needs a name to join

del plot['genre']

plot = plot.join(s)

plot.reset_index()



plot1 = pd.DataFrame(plot['genre'][plot['score']>=8.5])

plot2 = pd.DataFrame(plot['genre'][plot['score']<8.5])

plot1 = pd.DataFrame(data=plot1.groupby(['genre']).size())

plot2 = pd.DataFrame(data=plot2.groupby(['genre']).size())

plot1 = pd.concat([plot1, plot2], axis=1)

plot1.columns = ['High score', 'Not high score']

plot1 = plot1.replace(to_replace='NaN', value=0)

plot1.loc[:,:] = plot1.loc[:,:].div(plot1.sum(axis=1), axis=0)

plot1 = plot1.sort_values('High score',ascending=0)

del plot1['Not high score']

plot1[:20].plot(kind='bar', cmap='viridis', stacked=True)
plot = pd.DataFrame(data=d[["editors_choice","score","genre"]])

s = plot['genre'].str.split(', ').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1) # to line up with df's index

s.name = 'genre' # needs a name to join

del plot['genre']

plot = plot.join(s)

plot.reset_index()



plot1 = pd.DataFrame(plot['genre'][plot['score']>=8.5])

plot1 = pd.DataFrame(data=plot1.groupby(['genre']).size())



plot2 = pd.DataFrame(plot['genre'][plot['editors_choice']=='Y'])

plot2 = pd.DataFrame(data=plot2.groupby(['genre']).size())



plot1 = pd.concat([plot1, plot2], axis=1)

plot1.columns = ['High score', 'Editors choice']

plot1 = plot1.replace(to_replace='NaN', value=0)

plot = pd.DataFrame(abs(plot1['High score']-plot1['Editors choice'])/(plot1['High score']+plot1['Editors choice']))

plot = plot.sort_values(0, ascending= True)

plot.columns = ['Editors choice inconvinience']

plot[:20].plot(kind='bar', cmap='viridis', stacked=False)
import seaborn as sns

plt.xlim(1995,2017)

plt.ylim(1.8,10)

sns.kdeplot(d.release_year, d.score, n_levels=20, cmap="viridis", shade=1, shade_lowest=1)
genres = pd.DataFrame(d['genre'])

s = genres['genre'].str.split(', ').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1) # to line up with df's index

s.name = 'genre' # needs a name to join

del genres['genre']

genres = genres.join(s)

genres.reset_index()

genres_count=genres.groupby('genre').size()

large_genres=genres_count[genres_count>=500]

large_genres.sort_values(ascending=False,inplace=True)

data_genre = d[d.genre.isin(large_genres.keys())]

table_score = pd.pivot_table(data_genre,values=['score'],index=['release_year'],columns=['genre'],aggfunc='mean',margins=False)

table_count = pd.pivot_table(data_genre,values=['score'],index=['release_year'],columns=['genre'],aggfunc='count',margins=False)

table = table_score[table_count>10]

plt.figure()

sns.heatmap(table.score,linewidths=.5,annot=False,cmap='viridis')

plt.title('Average scores of games')
table_count.loc[:,:] = table_count.loc[:,:].div(table_count.sum(axis=0), axis=1)

sns.heatmap(table_count.score,linewidths=.5,annot=False,fmt='2.0f',vmin=0,cmap='viridis')

plt.title('Relative amount of games')
plot = pd.DataFrame(data=d["genre"])

s = plot['genre'].str.split(', ').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1) # to line up with df's index

s.name = 'genre' # needs a name to join

del plot['genre']

plot = plot.join(s)

plot.reset_index()



plot = pd.DataFrame(data=plot.groupby('genre').size())

plot = plot.replace(to_replace='NaN', value=0)

plot = plot.sort_values(0,ascending=0)

plot.columns = ['Games by genre']

plot['Reverse'] = plot['Games by genre'][0]

plot = plot.reset_index(drop=True)

plot.index = plot.index+1



i=1

while (i<=plot.count(0)['Reverse']):

    plot['Reverse'][i] = plot['Reverse'][i]/i

    i = i +1

plot.plot(cmap='viridis')
plot = pd.DataFrame(data=d["platform"])



plot = pd.DataFrame(data=plot.groupby('platform').size())

plot = plot.replace(to_replace='NaN', value=0)

plot = plot.sort_values(0,ascending=0)

plot.columns = ['Games by platform']

plot['Reverse'] = plot['Games by platform'][0]

plot = plot.reset_index(drop=True)

plot.index = plot.index+1



i=1

while (i<=plot.count(0)['Reverse']):

    plot['Reverse'][i] = plot['Reverse'][i]/i

    i = i +1

plot.plot(cmap='viridis')
plot = pd.DataFrame(data=d[["title","score"]])

plot['score'] = plot['score'].round()

plot['title'] = plot['title'].str.len()

plot = pd.DataFrame(data={'Title length':plot.groupby(['title','score']).size().index.get_level_values('title'),'Score':plot.groupby(['title','score']).size().index.get_level_values('score'),'Count':plot.groupby(['title','score']).size()})

plot = plot.pivot(index='Title length', columns='Score', values='Count')

plot = plot.replace(to_replace='NaN', value=0)

plot.loc[:,:] = plot.loc[:,:].div(plot.sum(axis=1), axis=0)



plot.plot(kind='area', cmap='viridis', stacked=True, ylim=[0,1])
import nltk

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

    return [(y,c) for y,c in without_stp.most_common(15)]





f, ax = plt.subplots(2,1, figsize=(4,6))



df = pd.DataFrame(d.title.unique())

df.columns = ['title']

t = df.title.apply(nltk.word_tokenize).sum()

without_stop = content_text(t)

df = pd.DataFrame(without_stop)

df.columns = ['Word','Appearances']

df.plot(kind='barh',x='Word',y='Appearances',cmap="viridis",ax=ax[0])



df = pd.DataFrame(d['title'][d['score']>=8.5].unique())

df.columns = ['title']

t = df.title.apply(nltk.word_tokenize).sum()

without_stop = content_text(t)

df = pd.DataFrame(without_stop)

df.columns = ['Word','Appearances']

df.plot(kind='barh',x='Word',y='Appearances',cmap="viridis",ax=ax[1])
master = d[d.score == 10][['title','platform','genre','release_year']]

f, ax = plt.subplots(2,1, figsize=(4,8))

master.groupby('platform').size().plot.pie(ax=ax[1],cmap='viridis')

s = master['genre'].str.split(', ').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1) # to line up with df's index

s.name = 'genre' # needs a name to join

del master['genre']

master = master.join(s)

master.reset_index()

master.groupby('genre').size().plot.pie(ax=ax[0],cmap='viridis')

ax[0].set_ylabel('')

ax[0].set_title('Count by genre')

ax[1].set_ylabel('')

ax[1].set_title('Count by platform')
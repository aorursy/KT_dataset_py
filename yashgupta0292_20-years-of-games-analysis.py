# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))







# Any results you write to the current directory are saved as output.
games=pd.read_csv('../input/ign.csv')
games.head()
games.isnull().sum()
games.drop(['Unnamed: 0','url'],axis=1,inplace=True) #dropping the unneeded values

games.drop(games.index[516],inplace=True) #dropping the game released on 1970. It looked to be an outlier
games.head(2)
games.isnull().sum()
games.dtypes
games['genre'].value_counts()[:10].plot(kind='pie',autopct='%1.1f%%')

plt.title('Distribution Of Top Genre"s')

plt.show()
from wordcloud import WordCloud, STOPWORDS



wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1200,

                          height=1000

                         ).generate(" ".join(games['title']))





plt.imshow(wordcloud)

plt.axis('off')

plt.show()
games.groupby('release_day')['genre'].count().plot(color='y')
fig,ax=plt.subplots(1,2,figsize=(18,10))

sns.countplot(games['release_month'],ax=ax[0],palette='Set1').set_title('Releases On Months')

plt.ylabel('')

sns.countplot(games['release_year'],ax=ax[1],palette='Set1').set_title('Releases on Years')

plt.xticks(rotation=90)

plt.show()
games['score'].hist(edgecolor='black')

plt.axvline(games['score'].mean(),color='b',linestyle='dashed')
games['platform'].value_counts()[:10].plot.pie(autopct='%1.1f%%')

plt.title('Top Platforms For Games')
plt.subplots(figsize=(15,15))

max_genres=games.groupby('genre')['genre'].count()

max_genres=max_genres[max_genres.values>200]

max_genres.sort_values(ascending=True,inplace=True)

mean_games=games[games['genre'].isin(max_genres.index)]

abc=mean_games.groupby(['release_year','genre'])['score'].mean().reset_index()

abc=abc.pivot('release_year','genre','score')

sns.heatmap(abc,annot=True,cmap='RdYlGn',linewidths=0.4)

plt.title('Average Score By Genre"s')

plt.show()



clone=games.copy()

clone['score_phrase']=clone['score_phrase'].map({'Awful':'Flop','Bad':'Flop','Disaster':'Flop','Unbearable':'Flop','Painful':'Flop','Mediocre':'Good','Okay':'Good','Great':'Hit','Amazing':'Hit','Masterpiece':'Masterpiece','Good':'Good'})

clone=clone[['score_phrase','platform','genre','score']]

max_platforms=clone['platform'].value_counts().index[:10]

plat=clone[clone['platform'].isin(max_platforms)]

plat=plat.groupby(['platform','score_phrase'])['score'].count().reset_index()

plat=plat.pivot('platform','score_phrase','score')

plat.plot.barh(width=0.9)

fig=plt.gcf()

fig.set_size_inches(12,14)

plt.show()
new_genres=max_genres.sort_values(ascending=False)[:10]

top_genres=games[games['genre'].isin(new_genres[:10].index)]

top_genres=top_genres.groupby(['release_year','genre'])['score'].count().reset_index()

top_genres=top_genres.pivot('release_year','genre','score')

sns.heatmap(top_genres,annot=True,fmt='2.0f',cmap='RdYlGn',linewidths=0.4)

fig=plt.gcf()

fig.set_size_inches(15,15)

plt.show()
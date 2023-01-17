import pandas as pd

import numpy as ny

import matplotlib.pyplot as plt

%matplotlib inline

data = pd.read_csv('../input/movie_metadata.csv')
data.shape
for i in data.columns:

    print (i, end= '; ')
Color_Count = data.color.value_counts()

plt.figure(1,figsize=(6,6))

idx = range(2)

labels = ['Color','Black & White']

plt.bar(idx,Color_Count,width=0.3)

plt.xticks(idx,labels)

plt.show()
Director = data.director_name.value_counts()

D_Name = Director.head(n=10).index

New_D = data[(data['director_name'].isin(D_Name))]

New_D.pivot_table(index=['director_name','imdb_score'],aggfunc='mean')



plt.figure(1,figsize=(12,6))

plt.subplot(1,2,1)

Director.head(n=10).sort_index().plot(kind='bar')

plt.title('Top 10 directors that have most volume movies')



plt.subplot(1,2,2)

New_D.groupby(['director_name'])['imdb_score'].mean().plot(kind='bar')

plt.xlabel("")

plt.title("Top 10 direcotors' average IMDB scores")



plt.show()
Language = data.language.value_counts()

Language.head(n=10).plot(kind='bar')

plt.title('Top 10 movie languages')

plt.show()
Country = data.country.value_counts()

Country.head(n=10).plot(kind='barh')

plt.title('Top 10 Countries that produce movies')

plt.show()
score_by_content = data.pivot_table(index=['content_rating'],values='imdb_score',aggfunc='mean')

Contents = data.content_rating.value_counts().sort_index()

plt.figure(1,figsize=(12,6))

plt.subplot(1,2,1)

plt.ylabel('Score')

plt.title('Average IMDB Socre by Movie Content')

score_by_content.plot(kind='bar')

plt.xlabel('')

plt.subplot(1,2,2)

Contents.plot(kind='bar')

plt.xlabel('Contents')

plt.ylabel('Volume')

plt.title('Movie amounts by content')

plt.show()
Year = data.title_year.value_counts().sort_index().tail(50)

year = range(50)

plt.figure(1,figsize=(12,6))

loc = range(3,49,5)

ticks = range(1970,2017,5)

plt.bar(year,Year)

plt.xticks(loc,ticks)

plt.xlabel('Year')

plt.title('Number of movies titled in recent 50 years',fontsize=15)

plt.show()
Gen = data['genres'].str.split('|')

New_Gen = []

Gen_Dict = {}

for item in Gen:

    for i in item:

        New_Gen.append(i)

        if i not in Gen_Dict:

            Gen_Dict[i] = 1

        else: Gen_Dict[i] += 1



Gen = pd.DataFrame.from_dict(Gen_Dict,orient='index')

Gen.columns = ['Counts']

Gen = Gen.sort_values('Counts',ascending=1)

Gen.plot(kind='barh',legend=False,figsize=(12,6))

plt.title('Movie amounts by different genres')

plt.show()
from os import path

from wordcloud import WordCloud,STOPWORDS

temp = pd.DataFrame(New_Gen)

text = temp.to_json()



wc = WordCloud(stopwords=STOPWORDS).generate(text)



plt.imshow(wc)

plt.axis("off")
Gen.index
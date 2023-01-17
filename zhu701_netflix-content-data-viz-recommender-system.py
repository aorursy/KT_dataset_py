import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot



df = pd.read_csv("../input/netflix-shows/netflix_titles.csv")



df.head()
types = df['type'].value_counts().reset_index()

types = types.rename(columns = {'type' : "count", "index" : 'type'})

fig = go.Figure(data=[go.Pie(labels=types['type'], values=types['count'], hole=.3,textinfo='label+percent')],

                layout=go.Layout(title="Content Type"))

fig.update_traces(marker=dict(colors=['gold', 'Indigo']))

fig.show()
df["date_added"] = pd.to_datetime(df['date_added'])

df['year'] = df['date_added'].dt.year



movie = df[df["type"] == "Movie"]

tv = df[df["type"] == "TV Show"]



movie = movie['year'].value_counts().reset_index()

movie = movie.rename(columns = {'year' : "count", "index" : 'year'})

movie = movie.sort_values('year')



tv = tv['year'].value_counts().reset_index()

tv = tv.rename(columns = {'year' : "count", "index" : 'year'})

tv = tv.sort_values('year')





fig = go.Figure(data=[go.Scatter(x=movie['year'], y=movie["count"], name="Movies",marker=dict(color='gold')),

                      go.Scatter(x=tv['year'], y=tv["count"], name="TV Shows",marker=dict(color='Indigo'))], 

                layout=go.Layout(title="Content Added Over Years"))

fig.show()
movie = df[df["type"] == "Movie"]



from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator

plt.rcParams['figure.figsize'] = (10, 10)

wordcloud = WordCloud(stopwords=STOPWORDS, background_color = 'black', width = 500,  height = 500, 

                      max_words = 100).generate(' '.join(movie['title'].str.lower()))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Words in Movie Title',fontsize = 30)

plt.show()
movie_c = movie['country'].value_counts().reset_index()

movie_c = movie_c.rename(columns = {'country' : "count", "index" : 'country'})

movie_c = movie_c.head(10)



fig = go.Figure(go.Treemap(

    labels = movie_c['country'],

    values = movie_c["count"], parents=["","", "", "", "", "", "", "", "", "",],

    textinfo = "label+value"), layout=go.Layout(title="Top 10 Countries with Most Movies"))



fig.show()
from collections import Counter

director_split = ", ".join(movie['director'].fillna("missing")).split(", ")

director_split = Counter(director_split).most_common(11)

del director_split[0]

fig = go.Figure(data=[go.Bar(y=[_[0] for _ in director_split][::-1], 

                             x=[_[1] for _ in director_split][::-1], orientation='h',marker=dict(color='DarkTurquoise'))], 

                layout=go.Layout(title="Director with Most Movies"))

fig.show()
movie_y = movie['release_year'].value_counts().reset_index()

movie_y = movie_y.rename(columns = {'release_year' : "count", "index" : 'release_year'})

movie_y = movie_y.sort_values('release_year')



fig = go.Figure(data=[go.Bar(x=movie_y['release_year'], y=movie_y["count"],marker=dict(color='Coral'))], 

                layout=go.Layout(title="Movie by Release Year"))

fig.show()
old_movie = movie.sort_values("release_year", ascending = True)

old_movie = old_movie[old_movie['duration'] != ""]

old_movie = old_movie[["title","country","release_year","rating","listed_in"]][:10]



fig = go.Figure(data=[go.Table(

    header=dict(values=list(old_movie.columns),

                fill_color='gold',

                align='left'),

    cells=dict(values=[old_movie.title, old_movie.country, old_movie.release_year, old_movie.rating, old_movie.listed_in],

               fill_color='white',

               align='left'))],layout=go.Layout(title="10 Oldest Movies"))

fig.show()
movie_r = movie['rating'].value_counts().reset_index()

movie_r = movie_r.rename(columns = {'rating' : "count", "index" : 'rating'})



fig = go.Figure(data=[go.Bar(x=movie_r['rating'], y=movie_r['count'])], 

                layout=go.Layout(title="Movies by Guideline Group"))

fig.update_layout(xaxis={'categoryorder':'total descending'})
genre_split = ", ".join(movie['listed_in']).split(", ")

genre_split = Counter(genre_split).most_common(20)



fig = go.Figure(data=[go.Pie(labels=[_[0] for _ in genre_split][::-1], values=[_[1] for _ in genre_split][::-1], 

                textinfo='label')], layout=go.Layout(title="Movies by Genre"))

fig.show()
imdb_ratings=pd.read_csv('../input/imdb-extensive-dataset/IMDb ratings.csv')

imdb_titles=pd.read_csv('../input/imdb-extensive-dataset/IMDb movies.csv')

ratings = pd.DataFrame({'Title':imdb_titles.title,

                    'Rating': imdb_ratings.weighted_average_vote})

ratings.drop_duplicates(subset=['Title','Rating'], inplace=True)

ratings.dropna()

join=ratings.merge(movie,left_on='Title',right_on='title',how='inner')

join=join.sort_values(by='Rating', ascending=False)



import plotly.express as px

top_rated=join[0:10]

fig =px.sunburst(

    top_rated,

    path=['title','country'],

    values='Rating',

    color='Rating')

fig.show()
join_drama = join[join["listed_in"].str.contains("Dramas")]

top_rated=join_drama[0:15]

top_rated = top_rated[["title","country","release_year","director","Rating"]]



fig = go.Figure(data=[go.Table(

    header=dict(values=list(top_rated.columns),

                fill_color='pink',

                align='left'),

    cells=dict(values=[top_rated.title, top_rated.country, top_rated.release_year, top_rated.director, top_rated.Rating],

               fill_color='white',

               align='left'))],layout=go.Layout(title="Top Rated Drama Movies"))

fig.show()
join_action = join[join["listed_in"].str.contains("Action")]

top_rated=join_action[0:15]

top_rated = top_rated[["title","country","release_year","director","Rating"]]



fig = go.Figure(data=[go.Table(

    header=dict(values=list(top_rated.columns),

                fill_color='orange',

                align='left'),

    cells=dict(values=[top_rated.title, top_rated.country, top_rated.release_year, top_rated.director, top_rated.Rating],

               fill_color='white',

               align='left'))],layout=go.Layout(title="Top Rated Action & Adventure Movies"))

fig.show()
join_thriller = join[join["listed_in"].str.contains("Thriller")]

top_rated=join_thriller[0:15]

top_rated = top_rated[["title","country","release_year","director","Rating"]]



fig = go.Figure(data=[go.Table(

    header=dict(values=list(top_rated.columns),

                fill_color='lightblue',

                align='left'),

    cells=dict(values=[top_rated.title, top_rated.country, top_rated.release_year, top_rated.director, top_rated.Rating],

               fill_color='white',

               align='left'))],layout=go.Layout(title="Top Rated Thriller Movies"))

fig.show()
def ratecountry(name):

    join_c = join[join["country"].fillna('missing').str.contains(name)]

    top_rated=join_c[0:10]

    trace = go.Bar(y=top_rated["title"], x=top_rated['Rating'], orientation="h", 

                   marker=dict(color="purple"))

    return trace



from plotly.subplots import make_subplots

traces = []

titles = ["United Kingdom","","Canada","","Spain"]

for title in titles:

    if title != "":

        traces.append(ratecountry(title))



fig = make_subplots(rows=1, cols=5, subplot_titles=titles)

fig.add_trace(traces[0], 1,1)

fig.add_trace(traces[1], 1,3)

fig.add_trace(traces[2], 1,5)



fig.update_layout(height=500, showlegend=False, yaxis={'categoryorder':'total ascending'})

fig.show()
tv = df[df["type"] == "TV Show"]



plt.rcParams['figure.figsize'] = (10, 10)

wordcloud = WordCloud(stopwords=STOPWORDS, background_color = 'black', width = 500,  height = 500, 

                      max_words = 100).generate(' '.join(tv['title'].str.lower()))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Words in TV Show Title',fontsize = 30)

plt.show()
tv_c = tv['country'].value_counts().reset_index()

tv_c = tv_c.rename(columns = {'country' : "count", "index" : 'country'})

tv_c = tv_c.head(10)



fig = go.Figure(go.Treemap(

    labels = tv_c['country'],

    values = tv_c["count"], parents=["","", "", "", "", "", "", "", "", "",],

    textinfo = "label+value"), layout=go.Layout(title="Top 10 Countries with Most TV Shows"))



fig.show()
def genrecountry(c):

    c = tv[tv["country"].fillna('missing').str.contains(c)]

    genre_split = ", ".join(c['listed_in']).split(", ")

    genre_split = Counter(genre_split).most_common(10)

    genre_name = [_[0] for _ in genre_split][::-1]

    genre_count = values=[_[1] for _ in genre_split][::-1]

    trace = go.Bar(y=genre_name, x=genre_count, orientation="h", 

                   marker=dict(color="Indigo"))

    return trace



traces = []

clist = ["United Kingdom","","Japan","","South Korea"]

for c in clist:

    if c != "":

        traces.append(genrecountry(c))



fig = make_subplots(rows=1, cols=5, subplot_titles=clist)

fig.add_trace(traces[0], 1,1)

fig.add_trace(traces[1], 1,3)

fig.add_trace(traces[2], 1,5)



fig.update_layout(height=500, showlegend=False, yaxis={'categoryorder':'total ascending'},title="Top 10 TV Show Genre from Each Country")

fig.show()
tv_d = tv['duration'].value_counts().reset_index()

tv_d = tv_d.rename(columns = {'duration' : "count", "index" : 'duration'})



fig = go.Figure(data=[go.Bar(x=tv_d['duration'], y=tv_d['count'],marker=dict(color='Darkgreen'))], 

                layout=go.Layout(title="TV Shows by Seasons"))

fig.update_layout(xaxis={'categoryorder':'total descending'})

fig.show()
manyseason = ['10 Seasons','11 Seasons','12 Seasons','13 Seasons','14 Seasons']

tv_long = tv[tv['duration'].isin(manyseason)]



tv_long = tv_long[["title","country","release_year","director","duration","listed_in"]]



fig = go.Figure(data=[go.Table(

    header=dict(values=list(tv_long.columns),

                fill_color='lightgreen',

                align='left'),

    cells=dict(values=[tv_long.title, tv_long.country, tv_long.release_year, tv_long.director, 

                       tv_long.duration, tv_long.listed_in],

               fill_color='white',

               align='left'))],layout=go.Layout(title="Longest TV Shows"))

fig.show()
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer



df2 = df[['title','type','director','rating','listed_in','description']]

df2.head()

df2['description'] = df2['description'].fillna('')

df2['director'] = df2['director'].fillna('')

df2['rating'] = df2['rating'].fillna('')

df2['listed_in'] = df2['listed_in'].map(lambda x: x.lower().split(','))

df2.set_index('title', inplace = True)



df2['Key_words'] = ''

columns = df2.columns

for index, row in df2.iterrows():

    words = ''

    for col in columns:

        words = words + ''.join(row[col])+ ' '

    row['Key_words'] = words

    

df2.drop(columns = [col for col in df2.columns if col!= 'Key_words'], inplace = True)
# instantiating and generating the count matrix

count = CountVectorizer()

count_matrix = count.fit_transform(df2['Key_words'])



# generating the cosine similarity matrix

cosine_sim = cosine_similarity(count_matrix, count_matrix)



indices = pd.Series(df2.index)

indices[:5]



# returning 10 recommended movies 

def recommendations(title, cosine_sim = cosine_sim):

    recommended_movies = []

    idx = indices[indices == title].index[0]

    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    top_10_indexes = list(score_series.iloc[1:11].index)

    for i in top_10_indexes:

        recommended_movies.append(list(df2.index)[i])

    return recommended_movies
recommendations('The Lord of the Rings: The Return of the King')
recommendations('I Am Mother')
import numpy as np 

import pandas as pd 

import json

from ast import literal_eval

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from os import path

from PIL import Image

% matplotlib inline
# carregando os datasets

df_movies = pd.read_csv("../input/tmdb_5000_movies.csv")

df_credits = pd.read_csv("../input/tmdb_5000_credits.csv")
# explorando os dados dos filmes

df_movies.head()
# Explorando os dados dos créditos dos filmes

df_credits.head()
# Renomeia a coluna id

df_credits.columns = ['id', 'title', 'cast', 'crew']



# Exclui a coluna 'title' que está repetida nas duas bases

df_credits.drop(columns=['title'], inplace=True)



# junta as duas bases em uma só, pelo ID

df_movies = df_movies.merge(df_credits, on='id')

df_movies.head()
# forma do novo dataframe

df_movies.shape
# colunas do novo dataframe

df_movies.dtypes
# converte o tipo de dados da coluna status para category

df_movies.status = df_movies.status.astype('category')

df_movies.status.dtype
# converte o tipo de dados das colunas de data

df_movies.release_date = pd.to_datetime(df_movies.release_date)
# Explorando os dados de colunas com valores em formato json

df_movies['genres'][0]
df_movies['keywords'][0]
df_movies['production_companies'][0]
df_movies['production_countries'][0]
df_movies['spoken_languages'][0]
# Faz o parse dos campos definidos em features para os seus correspondentes em python

features = ['cast', 'crew', 'keywords', 'genres', 'production_companies', 'spoken_languages', 'production_countries']

for feature in features:

    df_movies[feature] = df_movies[feature].apply(literal_eval)

    

# Método que recupera o diretor do filme

def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan



# Retorna uma lista dos primeiros n elementos (informado por parâmetro) ou a lista inteira se for menor

# Argumento 1: nome do campo, valor padrão: name

# Argumento 2: máximo de elementos a serem retornados, valor padrão: 5

def get_list(x, *args):

    if len(args) == 0:

        field = 'name'

        max_list = 5

    elif len(args) == 1:

        field = args[0]

        max_list = 5

    elif len(args) == 2:

        field = args[0]

        max_list = args[1]

        

    if isinstance(x, list):

        names = [i[field] for i in x]

        if len(names) > max_list:

            names = names[:max_list]

        return names



    # Retorna uma lista vazia em caso de falta dos dados ou de dados mal formados

    return []
# Acrescenta uma coluna com o nome do diretor 

df_movies['director'] = df_movies['crew'].apply(get_director)
# Acrescenta uma coluna com o ano de lançamento

df_movies['release_year'] = df_movies['release_date'].dt.year
# Converte o campos genres, keywords, production_companies, spoken_languages e production_countries em listas

df_movies['genres'] = df_movies['genres'].apply(get_list)

df_movies['keywords'] = df_movies['keywords'].apply(get_list, args=('name',))

df_movies['production_companies'] = df_movies['production_companies'].apply(get_list, args=('name',))

df_movies['spoken_languages'] = df_movies['spoken_languages'].apply(get_list, args=('iso_639_1',))

df_movies['production_countries'] = df_movies['production_countries'].apply(get_list, args=('name',))

df_movies['cast'] = df_movies['cast'].apply(get_list, args=('name', 10,))
df_movies.describe()
# Relação entre orçamento e receita

sns.pairplot(df_movies[['budget', 'revenue']])
sns.regplot(x="revenue", y="budget", data=df_movies);
df_movies['director'].value_counts().nlargest(10)
# Gera uma string separada por virgula das keywords dos filmes

words = []

df_movies['keywords'].apply(lambda x: words.extend(x))

words = ','.join(words)
# Gera uma nuvem de palavras com as keywords mais utilizadas

words_sep = words.replace(' ', '~')

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", width=800, height=400).generate(words_sep)



# Display the generated image:

plt.figure( figsize=(17,7) )

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
common_keywords = Counter(words.split(',')).most_common(30)

df_common_keywords = pd.DataFrame(common_keywords, columns=['keyword', 'counter'])

plt.figure( figsize=(20,5) )

barplot = sns.barplot(x=df_common_keywords['keyword'], y=df_common_keywords['counter'])

barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)

plt.show()
words = []

df_movies['cast'].apply(lambda x: words.extend(x))

actors_counter = Counter(words).most_common(10)

df_actors_counter = pd.DataFrame(actors_counter, columns=['name', 'counter'])

df_actors_counter
plt.figure( figsize=(20,5) )

sns.barplot(x=df_actors_counter['name'], y=df_actors_counter['counter'])

#barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)

#plt.show()
sns.pairplot(df_movies[['popularity', 'revenue']])
sns.scatterplot(x=df_movies['runtime'], y=df_movies['popularity'])
sns.scatterplot(x=df_movies['runtime'], y=df_movies['budget'])
# Pega os 30 filmes com popularidade mais alta

df_popular = df_movies.sort_values(by='popularity', ascending=False)[:30]



# Coloca todos os gêneros de todos os 30 filmes separados por vírgula

words = []

df_popular['genres'].apply(lambda x: words.extend(x))

words = ','.join(words)



# Conta os gêneros na string e monta um gráfico 

common_genres = Counter(words.split(',')).most_common()

df_common_genres = pd.DataFrame(common_genres, columns=['genre', 'counter'])

plt.figure( figsize=(20,5) )

barplot = sns.barplot(x=df_common_genres['genre'], y=df_common_genres['counter'])

barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)

plt.show()
words = []

df_movies['production_companies'].apply(lambda x: words.extend(x))

words = ','.join(words)



common_producers = Counter(words.split(',')).most_common(10)

df_common_producers = pd.DataFrame(common_producers, columns=['producer', 'counter'])

plt.figure( figsize=(20,5) )

barplot = sns.barplot(x=df_common_producers['producer'], y=df_common_producers['counter'])

barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)

plt.show()
words = []

df_movies['spoken_languages'].apply(lambda x: words.extend(x))

words = ','.join(words)



common_languages = Counter(words.split(',')).most_common(10)

df_common_languages = pd.DataFrame(common_languages, columns=['language', 'counter'])

plt.figure( figsize=(20,5) )

barplot = sns.barplot(x=df_common_languages['language'], y=df_common_languages['counter'])

plt.show()
words = []

df_movies['production_countries'].apply(lambda x: words.extend(x))

words = ','.join(words)



common_countries = Counter(words.split(',')).most_common(10)

df_common_countries = pd.DataFrame(common_countries, columns=['country', 'counter'])

plt.figure( figsize=(20,5) )

barplot = sns.barplot(x=df_common_countries['country'], y=df_common_countries['counter'])

plt.show()
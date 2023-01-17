import pandas as pd

import numpy as np

from ast import literal_eval

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

%matplotlib inline
movies = pd.read_csv("../input/the-movies-dataset/movies_metadata.csv")

print(movies.columns)

movies.head()
movies['genres'][0]
movies['genres'] = movies['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
keywords = pd.read_csv('../input/the-movies-dataset/keywords.csv')

keywords.head()
def clean_ids(x):

    try:

        return int(x)

    except:

        return np.nan



movies['id'] = movies['id'].apply(clean_ids)

movies = movies[movies['id'].notnull()]
movies['id'] = movies['id'].astype('int')

keywords['id'] = keywords['id'].astype('int')



movies = movies.merge(keywords, on='id')



movies.head()
movies["keywords"][0]
movies["keywords"] = movies["keywords"].apply(literal_eval)
def generate_list(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        if len(names) > 10:

            names = names[:10]

        return names

    return []



movies['keywords'] = movies['keywords'].apply(generate_list)

movies['genres'] = movies['genres'].apply(lambda x: x[:10])



movies[['title', 'keywords', 'genres']].head()
def sanitize(x):

    if isinstance(x, list):

        return [str.lower(i.replace(' ','')) for i in x]

    else:

        if isinstance(x, str):

            return str.lower(x.replace(' ', ''))

        else:

            return ''



for feature in ['genres', 'keywords']:

    movies[feature] = movies[feature].apply(sanitize)



def movie_soup(x):

    return  x["title"] + " " + " ".join(x['genres']) + " "+x['overview']+" "+" ".join(x['keywords'])



movies['overview'] = movies['overview'].fillna('')

movies['title'] = movies['title'].fillna('')

movies['soup'] = movies.apply(movie_soup, axis=1)
movies.loc[movies['title']=="The Matrix",'soup'].values
books = pd.read_csv("../input/top2k-books-with-descriptions/top2k_book_descriptions.csv", index_col=0)

print(books.columns)

books.head()
books['tag_name'][1]
books['tag_name'] = books['tag_name'].apply(lambda x: literal_eval(x) if literal_eval(x) else np.nan)

books = books[books['description'].notnull() | books['tag_name'].notnull()]

books = books.fillna('')
def book_soup(x):

    soup = x["original_title"]+" "+x["description"]+" "+" ".join(x['tag_name'])+" "+x["authors"]

    return soup
books["soup"] = books.apply(book_soup, axis=1)


soups = pd.concat([movies['soup'],books['soup']],ignore_index=True)




count = CountVectorizer(stop_words = "english")

count.fit(soups)



movies_matrix = count.transform(movies['soup'])

books_matrix = count.transform(books['soup'])



books_matrix.shape, movies_matrix.shape
cosine_sim = cosine_similarity(movies_matrix, books_matrix)
movies = movies.reset_index()

indices = pd.Series(movies.index, index=movies['title'].apply(lambda x: x.lower() if x is not np.nan else "")).drop_duplicates()
def content_recommender(title):

    idx = indices[title.lower()]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)

    

    sim_scores = sim_scores[:10]



    book_indices = [i[0] for i in sim_scores]



    return books.iloc[book_indices]




!pip3 install -q ipywidgets

!jupyter nbextension enable --py --sys-prefix widgetsnbextension

import ipywidgets

from IPython.display import HTML

def showhtml(recommendations):

    html = ' '.join([f"""

     <div class="flip-card">

      <div class="flip-card-inner">

        <div class="flip-card-front">

          <img src="{recommendations.iloc[i]['image_url']}" alt="Avatar" style="width:300px;height:300px;">

        </div>

        <div class="flip-card-back">

          <h4>{recommendations.iloc[i]['title']}</h4>

          <p>by {recommendations.iloc[i]['authors']}</p>

        </div>

      </div>

    </div> """ for i in range(10)])

    html = "<div class='grid'>"+html+"</div>"

    html +="""<style>

    .flip-card {

      background-color: transparent;

      width: 200px;

      height: 300px;

      border: 1px solid #f1f1f1;

    }



    .flip-card-inner {

      position: relative;

      width: 100%;

      height: 100%;

      text-align: center;

      transition: transform 0.8s;

      transform-style: preserve-3d;

    }



    .flip-card:hover .flip-card-inner {

      transform: rotateY(180deg);

    }



    .flip-card-front, .flip-card-back {

      position: absolute;

      width: 100%;

      height: 100%;

      -webkit-backface-visibility: hidden; /* Safari */

      backface-visibility: hidden;

    }



    .flip-card-front {

      background-color: #bbb;

      color: black;

    }



    .flip-card-back {

    padding:10px;

      background-color: dodgerblue;

      color: white;

      transform: rotateY(180deg);

    }

    .grid {

        display: grid;

        grid-template-columns: 30% 30% 30%;

        grid-template-rows: 25% 25% 25%;

        grid-gap: 5%;

    }

    </style>"""

    return html





def show_books(movie_name='I, robot'):

    recommendations = content_recommender(movie_name)

#     for i in range(10):

#         disPic(recommendations["image_url"].iloc[i])

#         print(recommendations["original_title"].iloc[i])

#         print(recommendations["description"].iloc[i])

    display(HTML(showhtml(recommendations)))

display(ipywidgets.interact(show_books))



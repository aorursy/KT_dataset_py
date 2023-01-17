import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer

!pip install rake-nltk

from rake_nltk import Rake
dataset = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')

dataset[0:1]
feature = dataset[['Title','Genre','Director','Actors','Plot']]

feature.head()
feature['bag_of_word']= ''

for index,row in feature.iterrows():

    plot = row['Plot']

    r = Rake()

    r.extract_keywords_from_text(plot)

    keyword_score = r.get_word_degrees()

    g = ''.join(row['Genre'].split(',')).lower()

    d = ''.join(row['Director'].split(' ')).lower()

    a = ' '.join(row['Actors'].replace(' ','').split(',')).lower()

    k = ' '.join(list(keyword_score.keys()))

    row['bag_of_word'] = g + ' ' + ' ' + d + ' ' + a + ' ' + k



df = feature[['Title','bag_of_word']]

df.head()
c = CountVectorizer()

count_mat = c.fit_transform(df['bag_of_word'])
cosine_sim = cosine_similarity(count_mat,count_mat)

print(cosine_sim)
indices = pd.Series(df.Title)

def recommend_movie(title):

    movie=[]

    idx = indices[indices == title].index[0]

    sort_index = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    top_10 = sort_index.iloc[1:11]

    for i in top_10.index:

        movie.append(indices[i])

    return movie
recommend_movie('Guardians of the Galaxy Vol. 2')
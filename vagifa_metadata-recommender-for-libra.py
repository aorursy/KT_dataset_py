import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from string import punctuation
  

# Recommender function

def metadata_recommender(data,feature_names,search_name,n_recommendations=10,indexer='title'):

    

    # Basic data cleaning function for metadata-based recommender

    def clean_data(x):

        if isinstance(x, list):

            return [str.lower(i.replace(" ", "")) for i in x if not i.isdigit()]

        else:

            if isinstance(x, str):

                return str.lower(x.replace(" ", ""))

            else:

                return ''

    

    # Create recommender 'soup'

    def create_soup(x):

        soup = []

        for feature in feature_names:

            f = ''.join(x[feature])

            soup.append(f)

        return ' '.join(soup)

     

    # clean data iteratively

    for feature in feature_names:

        data[feature] = data[feature].apply(clean_data)

    

    # define the soup

    data['soup'] = data.apply(create_soup,axis=1)   

    count_vec = CountVectorizer()

    # BOW and similarity matrix

    count_matrix = count_vec.fit_transform(data['soup'])

    sim_matrix = cosine_similarity(count_matrix,count_matrix)

    

    # mapping for the results

    data = data.reset_index()

    mapping = pd.Series(data.index, index=data[indexer])

    

    # get n recommendations

    def extended_recommender():

        index = mapping[search_name]

        similarity_score = list(enumerate(sim_matrix[index]))

        try:

            similarity_score = sorted(similarity_score, key=lambda x: x[1],reverse=True)

        except:

            similarity_score = sorted(similarity_score, key=lambda x: x[0],reverse=True)

        similarity_score = similarity_score[1:n_recommendations]

        indices = [i[0] for i in similarity_score]

        return data[indexer].iloc[indices]

    

    return extended_recommender()
df = pd.read_csv('../input/disney-plus-shows/disney_plus_shows.csv')
metadata_recommender(df,['plot','genre','genre','director','writer'],'Coco')
import sqlite3



cnx = sqlite3.connect('../input/pitchfork-data/database.sqlite')



artists = pd.read_sql_query("SELECT * FROM artists", cnx)

content = pd.read_sql_query("SELECT * FROM content", cnx)

genres = pd.read_sql_query("SELECT * FROM genres", cnx)

labels = pd.read_sql_query("SELECT * FROM labels", cnx)

reviews = pd.read_sql_query("SELECT * FROM reviews", cnx)
merged = pd.merge(reviews,content,on='reviewid').merge(genres,on='reviewid').merge(labels,on='reviewid')
metadata_recommender(merged,['content','genre','label'],'mezzanine')
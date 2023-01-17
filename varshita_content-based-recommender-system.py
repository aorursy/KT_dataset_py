import pandas as pd

import numpy as np



from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import os

print(os.listdir("../input"))
#Read data into dataframe

songs = pd.read_csv('../input/songlyrics/songdata.csv')

# Inspect the data

songs.head()
songs.describe(include = 'all')
# resample only 5000 songs out of 57650 songs available

songs = songs.sample(n = 5000)
songs.describe(include = 'all')
# Replace \n present in the text

songs['text'] = songs['text'].str.replace('\n', '' )
# Initialize tfidf vectorizer

tfidf = TfidfVectorizer(analyzer = 'word', stop_words= 'english')



# this is used for calculating tf (term frequency) and idf (inverse document frequency)



#tf = number of occurence of that word/ total number of words

#idf = log(total number of documents/no of documents containg the term)



# now we are going to calculate the TF-IDF score for each song lyric word by word i.e. TF * IDF

#tfidf.get_feature_names()
# Fit and transform 

tfidf_matrix = tfidf.fit_transform(songs['text'])
tfidf_matrix.shape 

# from the shape, we see that the matrix has 

# as rows all the 5000 songs and each words 

# importance corresponding to the song is given
tfidf_matrix.todense()
# calculate similarity of one lyric to the next (using euclidean distance or cosine similarity)

# we use cosine because we are not only interested in the magnitude of the term frequency but also the angle between the lyrics; small angle means more similar

# (refer to http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/)

from sklearn.metrics.pairwise import cosine_similarity



#cosine_similarity(tfidf_matrix[0:1], tfidf_matrix) # to see the similarity of the first song with all the other songs; in the output, first value is 1 because the song is being compared to itself 

cosine_similarities = cosine_similarity(tfidf_matrix)
cosine_similarities[0].argsort()[:-50:-1] # argort gives you the position (starting from 0) rather than the value itself

# getting the column numbers which essentialy represent the song number of the 50 most similar songs to the song[0]

# creating a dictionary to store for each song 50 most similar song

my_dict = {}



for i in range(len(cosine_similarities)):

    similar_indices = cosine_similarities[i].argsort()[:-50:-1] # returns the indexes of top 50 songs

    

    # setting the key as the song name

    song_name = songs['song'].iloc[i] 

    # setting the value as three items i.e (1) similarity score, (2) song name, (3) artist name

    my_dict[song_name] = [(cosine_similarities[i][x], songs['song'].iloc[x], songs['artist'].iloc[x]) for x in similar_indices][1:]
# Testing whether this dictionary works ro give most similar songs for Song number 10

getforsong = songs['song'].iloc[10]

getforsong

my_dict.get(getforsong)[:3]
# Let us use this dictionary to present the recommendations



def get_recommendation(ques):

            # Get song to find recommendations for

            song = ques['song']

            # Get number of songs to recommend

            number_songs = ques['no_of_songs']

            # Get the number of songs most similars from matrix similarities

            recom_song = my_dict.get(song)[:number_songs]

            # print each item

            print(f"The recommended songs for {song} by {songs.loc[songs['song'] == song, 'artist'].iloc[0]} are")

            for i in range(number_songs):

                print(f"{recom_song[i][1]} by {recom_song[i][2]} with similarity score {recom_song[i][0]} ")

#             print(recom_song)
# Create a dictionary to pass as input



ques = {

    "song" : songs['song'].iloc[1104],

    "no_of_songs" : 4

}



get_recommendation(ques)
# to find the artist name given the song name

# songs.loc[songs['song'] == "Pieces Of A Dream", 'artist'].iloc[0]
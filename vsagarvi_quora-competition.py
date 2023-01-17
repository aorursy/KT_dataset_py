# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300/"))



# Any results you write to the current directory are saved as output.
import gensim

path = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings = gensim.models.KeyedVectors.load_word2vec_format(path, binary =True)
embeddings['amazon'] #for every word google has a vector representation
embeddings.most_similar('modi', topn=10) #shows terms related to hyundai
embeddings.doesnt_match(['rahul', 'modi', 'sonia', 'sachin']) #tells odd one out from list
embeddings.doesnt_match(['football', 'cricket', 'basketball', 'swimming'])
embeddings.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
url  = 'https://bit.ly/2G4zbHA'

amazon = pd.read_csv(url)

amazon.shape
docs = amazon['reviewText'].fillna('').str.lower()

docs = docs.str.replace('[^a-z]', ' ')

docs.head()
import nltk

stopwords = nltk.corpus.stopwords.words('english')

stemmer = nltk.stem.PorterStemmer()

def clean_sentence(text):

    words = text.split(' ')

    words_clean = [stemmer.stem(word) for word in words if word not in  stopwords]

    return ' '.join(words_clean)

docs_clean = docs.apply(clean_sentence)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df = 5)

vectorizer.fit(docs_clean)

df_dtm = pd.DataFrame(vectorizer.transform(docs_clean).toarray(),

                     columns = vectorizer.get_feature_names())

df_dtm.shape
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity([df_dtm['kindl'], df_dtm['book']])
cosine_similarity([embeddings['kindle'], embeddings['book']])
cos_matrix = pd.DataFrame(cosine_similarity(df_dtm.T), columns = df_dtm.columns, index = df_dtm.columns)

cos_matrix
cos_matrix['tablet'].sort_values(ascending=False).drop('tablet')
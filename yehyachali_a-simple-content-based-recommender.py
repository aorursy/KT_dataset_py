import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
%matplotlib inline
books = pd.read_csv("../input/top2k-books-with-descriptions/top2k_book_descriptions.csv", index_col=0)
books

books['tag_name'] = books['tag_name'].apply(lambda x: literal_eval(x) if literal_eval(x) else np.nan)
mabooks = books[books['description'].notnull() | books['tag_name'].notnull()]
mabooks = mabooks.fillna('')
def make_soup(x):
    soup = x["original_title"]+" "+x["description"]
    return soup
soups = mabooks.apply(make_soup, axis=1).rename("soup")
mabooks= mabooks.join(soups)
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(mabooks['soup'])

tfidf_matrix.shape



cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(mabooks.index, index=mabooks['original_title'].apply(lambda x: x.lower())).drop_duplicates()

def content_recommender(title, cosine_sim=cosine_sim, mabooks=mabooks, indices=indices):
    idx = indices[title.lower()]

    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    book_indices = [i[0] for i in sim_scores]

    return mabooks['title'].iloc[book_indices]

content_recommender('The Da Vinci Code')




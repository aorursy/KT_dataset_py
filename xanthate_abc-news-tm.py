import numpy as np

import pandas as pd



import os

osj = os.path.join



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(osj(dirname, filename))
%%time

fp = '/kaggle/input/million-headlines/abcnews-date-text.csv'

df = pd.read_csv(fp, parse_dates=[0], infer_datetime_format=True)

#df = df[:25000]

df.head()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation as LDA
# get a tuple of top n words and their counts across millions of headlines - (word, count)

def get_nwords(top_nwords, countvectorizer, text_data):

    vectorized_headlines = countvectorizer.fit_transform(text_data.values)

    # get the total number of times each word appeared - excluding stop words

    vectorized_total = np.sum(vectorized_headlines, axis=0)

    # get index of maximum values

    indicies = np.flip(np.argsort(vectorized_total)[0, :], 1)

    # get the maximum values

    values = np.flip(np.sort(vectorized_total)[0, :], 1)

    vectors = np.zeros((top_nwords, vectorized_headlines.shape[1]))

    

    for i in range(top_nwords):

        vectors[i, indicies[0, i]] = 1

        

    words = [word[0].encode('ascii').decode('utf-8') for word in countvectorizer.inverse_transform(vectors)]

    return (words, values[0, :top_nwords].tolist()[0])



countvectorizer = CountVectorizer(stop_words='english')



words, word_counts = get_nwords(top_nwords=15, countvectorizer=countvectorizer, text_data=df.headline_text)
import matplotlib.pyplot as plt

import scipy.stats as stats



fig, ax = plt.subplots(figsize=(16,8))

ax.bar(range(len(words)), word_counts);

ax.set_xticks(range(len(words)));

ax.set_xticklabels(words, rotation='vertical');

ax.set_title('Top words in headlines dataset (excluding stop words)');

ax.set_xlabel('Word');

ax.set_ylabel('Number of occurences');

plt.show()
corpus = ['Here comes the sun over the mountain in the east', 

          'Norwegian wood under the hood',

          'Sky comes the house, west and east',

          'Mickey mouse is in the house',

          'over the hill rises, in the east']



cv = CountVectorizer(stop_words='english')

X = cv.fit_transform(corpus)

cv.get_feature_names()
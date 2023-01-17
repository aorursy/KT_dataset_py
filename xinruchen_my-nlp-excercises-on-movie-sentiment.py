import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import gensim
import nltk
import sklearn

import re
import string
from string import digits

movie_train = pd.read_csv( "../input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
movie_test = pd.read_csv( "../input/testData.tsv", header=0, delimiter="\t", quoting=3 )

movie_train.columns
movie_train.head()
#text = movie_train['review'][0]
#text.replace('\\', ' ')
movie_test.head()
# Cleaning the dataset
def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'http\S+', '', elem))
    df[text_field] = df[text_field].apply(lambda elem: elem.replace('<br />', ' '))
    df[text_field] = df[text_field].apply(lambda elem: elem.replace('\\', ' '))
    df[text_field] = df[text_field].apply(lambda elem: elem.replace('_', ' '))
    
    remove_digits = str.maketrans('', '', digits)
    df[text_field] = df[text_field].apply(lambda elem: elem.translate(remove_digits))
    return df

clean_movie_train = clean_text(movie_train, 'review')
clean_movie_test = clean_text(movie_test, 'review')
# Tokenization
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

clean_movie_train['tokens'] = clean_movie_train['review'].apply(tokenizer.tokenize)
clean_movie_train.head()

clean_movie_test['tokens'] = clean_movie_test['review'].apply(tokenizer.tokenize)
# Explore words and sentences
all_words = [word for tokens in clean_movie_train['tokens'] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clean_movie_train['tokens']]
Vocab = sorted(list(set(all_words)))
print('%s words total, with a vocabulary size of %s' %(len(all_words), len(Vocab)))
# Explore vocabulary
print(Vocab[0:100])

from collections import Counter
count_all_words = Counter(all_words)
count_all_words.most_common(100)
#Embedding 1
# TFIDF bag of words
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', token_pattern = r'\w+')

tfidf = dict()
tfidf['all_train'] = (tfidf_vectorizer.fit_transform(clean_movie_train['review']))
tfidf['test'] = (tfidf_vectorizer.transform(clean_movie_test['review']))
print(tfidf['all_train'].shape)
print(tfidf['test'].shape)
#Embedding 2
# word2vec 
#from gensim.models import Word2Vec
#model = Word2Vec.load("300features_40minwords_10context")
# The Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

train_target = clean_movie_train['sentiment']
classifier.fit(tfidf['all_train'], train_target)

submission = pd.DataFrame.from_dict({'id': clean_movie_test['id']})
lr_predict = classifier.predict(tfidf['test'])

submission = pd.DataFrame(data={"id":clean_movie_test["id"], "sentiment":lr_predict})
submission.to_csv( "submission.csv", index=False, quoting=3 )
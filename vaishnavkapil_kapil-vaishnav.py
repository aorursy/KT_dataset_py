from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
dataset = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
dataset1 = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
corpus = []
for i in range(0, 27481):
    text_review = re.sub('[^a-zA-Z]',' ', str(dataset['text'][i]))+' '+str(dataset['sentiment'][i])
    text_review = text_review.lower()
    text_review = text_review.split()
    ps = PorterStemmer()
    text_review = [ps.stem(word) for word in text_review if not word in set(stopwords.words('english'))]
    text_review = ' '.join(text_review)
    corpus.append(text_review)
    
corpus1 = []
for j in range(0, 3534):
    text_review1 = re.sub('[^a-zA-Z]',' ', str(dataset1['text'][j]))+' '+str(dataset1['sentiment'][j])
    text_review1 = text_review1.lower()
    text_review1 = text_review1.split()
    ps = PorterStemmer()
    text_review1 = [ps.stem(word) for word in text_review1 if not word in set(stopwords.words('english'))]
    text_review1 = ' '.join(text_review1)
    corpus1.append(text_review1)
cv = CountVectorizer()
X_fit = (cv.fit(corpus))
X_train = X_fit.transform(corpus)
X_test = X_fit.transform(corpus1)
X_train_vect = pd.DataFrame(X_train.toarray())
X_test_vect = pd.DataFrame(X_test.toarray())
   
corpus2 = []  
for i in range(0, 27481):
    y_train = str(dataset['selected_text'][i])
    y_train = y_train.split()
    y_train = ' '.join(y_train)
    corpus2.append(y_train)
indexed_sentences = []

i = 2
word2idx = {'START': 0, 'END': 1}
for sentence in corpus2:
    indexed_sentence = []
    for token in sentence:
        token = token.lower()
        if token not in word2idx:
            word2idx[token] = i
            i += 1

        indexed_sentence.append(word2idx[token])
    indexed_sentences.append(indexed_sentence)

print("Vocab size:", i)
word_idx_count = {
    0: float('inf'),
    1: float('inf'),
  }
idx2word = ['START', 'END']
index_sentences = []
wordidx = {'START': 0, 'END': 1}
for sentence in corpus2:
    index_sentence = []
    for token in sentence:
        token = token.lower()
        if token not in wordidx:
            idx2word.append(token)
            wordidx[token] = i
            i += 1

      # keep track of counts for later sorting
        idx = wordidx[token]
        word_idx_count[idx] = word_idx_count.get(idx, 0) + 1

        index_sentence.append(idx)
    index_sentences.append(index_sentence)
Y=MultiLabelBinarizer().fit_transform(indexed_sentences)
#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators = 2, max_depth = None, n_jobs = -1)
#rf_model= rf.fit(X_train_vect, vk)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train_vect, Y)
selected_text = classifier.predict(X_test_vect)
idxword = dict((v, k) for k, v in iteritems(wordidx))
idxword[1]
def get_words(sentence_num):
    return ' '.join(idxword[z] for z in sentence)
sent = []
for sentence_num in indexed_sentences:
    word = get_words(sentence_num)
    sent.append(word)   


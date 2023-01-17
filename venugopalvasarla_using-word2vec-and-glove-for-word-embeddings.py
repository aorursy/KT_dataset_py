import re

import nltk

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')
#let's view the fileids of the gutenberg corpus

from nltk.corpus import gutenberg

print(gutenberg.fileids())
#lets choose the bryant stories

bryant_sents = gutenberg.raw(gutenberg.fileids()[5])

bryant_sents = bryant_sents.split('\n')

print('the lenght of the sentences before cleaning and preprocessing is', len(bryant_sents))
#an exmaple line in the sents

print(bryant_sents[7])
#importing necessary libraires to clean the data

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

stopWords = stopwords.words('english')

charfilter = re.compile('[a-zA-Z]+')
#now let's tokenize the words

def simple_filter(sent):

    #converting all the tokens to lower case:

    words = sent.split()

    word_lower = [] 

    for word in words:

        word_lower.append(word.lower())

    #let's remove every stopword:

    word_clean = [word for word in word_lower if word not in stopWords]

    #removing all the characters and using only characters

    tokens = list(filter(lambda token : charfilter.match(token),word_clean))

    #stemming all the words

    ntokens = []

    for word in tokens:

        ntokens.append(PorterStemmer().stem(word))

    return tokens
#converting all the bryant data to tokens using our function simple tokenizer we created earlier

sentences = []

for sent in bryant_sents:

    tokens = simple_filter(sent)

    if len(tokens) >0 :

        sentences.append(tokens)
#an example sentence in the data

print(sentences[7])
#Word2Vec

#training the gensim on the data

#Using the Cbow architecture for the word2Vec

from gensim.models import Word2Vec

model_cbow = Word2Vec(sentences, min_count = 1, size = 50, workers = 3, window = 5, sg = 0)
#Any example to find the vector model of a word

print('the array representation of the word \'gentleman\'\n:',model_cbow['gentleman'], '\n the array representation of the word \'messenger\'\n:', model_cbow['messenger']) 
#Computing the similarities of the words

print(model_cbow.similarity('messenger', 'gentleman'))
#Computing the 5 most similar words to the word gentleman

print('the 5 most similar words to \'gentleman\':', model_cbow.most_similar('gentleman')[:5])
# defining a tsne function to visualize

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

def plot_tsne(model, num):

    labels = []

    tokens = []

    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    tsne = TSNE(perplexity = 40, n_components = 2, init = 'pca', n_iter = 2500, random_state = 23)

    data = tsne.fit_transform(tokens[:num])

    x = []

    y = []

    for each in data:

        x.append(each[0])

        y.append(each[1])

    plt.figure(figsize = (10, 10))

    for i in range(num):

        plt.scatter(x[i], y[i])

        plt.annotate(labels[i],

                     xy = (x[i], y[i]),

                     xytext = (5,2),

                     textcoords = 'offset points',

                     ha = 'right',

                     va = 'bottom')

    plt.show()
#visualising the cbow archtecture(only the first 300)

plot_tsne(model_cbow, 300)
#let's see how the skipgram model works on the data

model_skipgram = Word2Vec(sentences, min_count = 1, size = 50, workers = 3, window = 5, sg = 1)
#Any example to find the vector model of a word

print('the array representation of the word \'gentleman\'\n:',model_skipgram['gentleman'], '\n the array representation of the word \'messenger\'\n:', model_skipgram['messenger']) 
#Computing the similarities of the words

print(model_skipgram.similarity('messenger', 'gentleman'))
#Computing the 5 most similar words to the word gentleman

print('the 5 most similar words to \'gentleman\':', model_skipgram.most_similar('gentleman')[:5])
#visualising the skipgram archtecture(only the first 300)

plot_tsne(model_skipgram,100)
#using the glove package for embeddings

!pip install glove_python
from glove import Corpus, Glove

corpus = Corpus()

corpus.fit(sentences, window = 5)

glove = Glove(no_components = 50, learning_rate = 0.05)

glove.fit(corpus.matrix, epochs = 30, no_threads = 4, verbose = True)

glove.add_dictionary(corpus.dictionary)
#Any example to find the vector model of a word

print('the array representation of the word \'gentleman\'\n:',glove.word_vectors[glove.dictionary['gentleman']],

      '\n the array representation of the word \'messenger\'\n:', glove.word_vectors[glove.dictionary['messenger']]) 
#Computing the similarities of the words

print(glove.most_similar('gentleman', number = 5))
# now visualising first 300 words using tsne

def plot_tsne_glove(model, num):

    labels = []

    tokens = []

    for word in model.wv.vocab:

        tokens.append(glove.word_vectors[glove.dictionary[word]])

        labels.append(word)

    tsne = TSNE(perplexity = 40, n_components = 2, init = 'pca', n_iter = 2500, random_state = 23)

    data = tsne.fit_transform(tokens[:num])

    x = []

    y = []

    for each in data:

        x.append(each[0])

        y.append(each[1])

    plt.figure(figsize = (10, 10))

    for i in range(num):

        plt.scatter(x[i], y[i])

        plt.annotate(labels[i],

                     xy = (x[i], y[i]),

                     xytext = (5,2),

                     textcoords = 'offset points',

                     ha = 'right',

                     va = 'bottom')

    plt.title('Word vectorization using Glove')

    plt.show()
plot_tsne_glove(model_skipgram, 300)
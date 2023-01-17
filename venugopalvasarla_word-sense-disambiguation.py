#let's get our data

import nltk

from nltk.corpus import webtext
print('the data contained in the webtext is :',webtext.fileids())
#let's use the firefox webchat 

firefox = webtext.raw('firefox.txt')

print(firefox[:500])
firefox_sents = firefox.split('\n')

print('the number of sentences converted from the raw data file is:', len(firefox_sents))
#cleaning the data

#libaries

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

Stopwords = stopwords.words('english')

import re

charfilter = re.compile('[a-zA-Z]+')
#now let's tokenize the words

def simple_filter(sent):

    #converting all the tokens to lower case:

    words = sent.split()

    word_lower = []

    for word in words:

        word_lower.append(word.lower())

    #let's remove every stopword:

    #word_clean = [word for word in word_lower if word not in Stopwords]

    #removing all the characters and using only characters

    tokens = list(filter(lambda token : charfilter.match(token),word_lower))

    #stemming all the words

    ntokens = []

    for word in tokens:

        ntokens.append(PorterStemmer().stem(word))

    return ' '.join(tokens)
#converting all the bryant data to tokens using our function simple tokenizer we created earlier

sentences = []

for sent in firefox_sents:

    tokens = simple_filter(sent)

    if len(tokens) >0 :

        sentences.append(tokens)
#Sense2Vec

#using Spacy to compute part of speech tags of every word

import spacy

nlp = spacy.load('en', disable = ['parser', 'ner'])
docs = []

count = 0

for item in sentences:

    docs.append(nlp(item))

    count += 1

sense_corpus = [[x.text+"_"+x.pos_ for x in y] for y in docs]
# see some of the data

print(sense_corpus[:5])
#Now using word2vec model on the sense corpus we created

from gensim.models import Word2Vec

model_skipgram = Word2Vec(sense_corpus, min_count = 1, size = 50, workers = 3, window = 5, sg = 1)
import warnings

warnings.filterwarnings('ignore')
#lets print a word from the vocabulary

print('the first word is:',model_skipgram[sense_corpus[0][0]])
print('the similarity between web and internet:', model_skipgram.similarity('web_NOUN', 'internet_NOUN'))
#the above code shows how well our sense2vec has performed on our dataset

#now let's try to see all the nearest data to web

print('the 5 most similar word to web are', model_skipgram.most_similar('web_NOUN')[:10])
# now let;s wrap it up and then visualize the data

#using TSNE

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

labels = []

tokens = []

for word in model_skipgram.wv.vocab:

    tokens.append(model_skipgram[word])

    labels.append(word)

tsne = TSNE(perplexity = 40, n_components = 2, init = 'pca', n_iter =  2500, random_state = 23)

data = tsne.fit_transform(tokens[:150])

x = []

y = []

for each in data:

    x.append(each[0])

    y.append(each[1])

plt.figure(figsize = (12, 12))

for i in range(150):

    plt.scatter(x[i], y[i])

    plt.annotate(labels[i],

                 xy = (x[i], y[i]),

                 xytext = (5,2),

                 textcoords = 'offset points',

                 ha = 'right',

                 va = 'bottom')

plt.title('Tsne visualization of the sense2vec')

plt.show()
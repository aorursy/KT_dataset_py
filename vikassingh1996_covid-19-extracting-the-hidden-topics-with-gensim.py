import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel



# spacy for lemmatization

import spacy



# Plotting tools

import pyLDAvis

import pyLDAvis.gensim  # don't skip this



# nltk

import re

import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
biorxiv = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")

biorxiv = biorxiv.fillna("No Information")

biorxiv.head()
stop_words = set(stopwords.words("english"))



def clean_text(s):

    words = str(s).lower()

    words = re.sub('\[.*?\]', '', words)

    words = re.sub('https?://\S+|www\.\S+', '', words)

    words = re.sub('<.*?>+', '', words)

    words = re.sub('[%s]' % re.escape(string.punctuation), '', words)

    words = re.sub('\n', '', words)

    words = re.sub('\w*\d\w*', '', words)

    words = word_tokenize(words)

    words = [w for w in words if not w in stop_words]

    words = [w for w in words if w.isalpha()]

    words =  ' '.join(words)

    return words



#source: https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove

def get_top_unigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(1, 1)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



def get_top_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



def get_top_threegrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



def get_top_fourgrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(4, 4)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
%%time

title = biorxiv['title'].apply(lambda x : clean_text(x))



plt.style.use('ggplot')

fig, axes = plt.subplots(2, 2, figsize=(18, 20), dpi=100)

           

top_unigrams=get_top_unigrams(title)[:20]

x,y=map(list,zip(*top_unigrams))

sns.barplot(x=y,y=x, ax=axes[0,0], color='dodgerblue')





top_bigrams=get_top_bigrams(title)[:20]

x,y=map(list,zip(*top_bigrams))

sns.barplot(x=y,y=x, ax=axes[0,1], color='orangered')



top_threegrams=get_top_threegrams(title)[:20]

x,y=map(list,zip(*top_threegrams))

sns.barplot(x=y,y=x, ax=axes[1, 0], color='limegreen')



top_fourgrams=get_top_fourgrams(title)[:20]

x,y=map(list,zip(*top_fourgrams))

sns.barplot(x=y,y=x, ax=axes[1, 1], color='red')





axes[0, 0].set_ylabel(' ')

axes[0, 1].set_ylabel(' ')

axes[1, 0].set_ylabel(' ')

axes[1, 1].set_ylabel(' ')



axes[0, 0].yaxis.set_tick_params(labelsize=15)

axes[0, 1].yaxis.set_tick_params(labelsize=15)

axes[1, 0].yaxis.set_tick_params(labelsize=15)

axes[1, 1].yaxis.set_tick_params(labelsize=15)



axes[0, 0].set_title('Top 20 most common unigrams in title', fontsize=15)

axes[0, 1].set_title('Top 20 most common bigrams in title', fontsize=15)

axes[1, 0].set_title('Top 20 most common threegrams in title', fontsize=15)

axes[1, 1].set_title('Top 20 most common fourgrams in title', fontsize=15)



plt.tight_layout()

plt.show()
abstract = biorxiv['abstract'].apply(lambda x : clean_text(x))



plt.style.use('ggplot')

fig, axes = plt.subplots(2, 2, figsize=(18, 20), dpi=100)

plt.tight_layout()



top_unigrams=get_top_unigrams(abstract)[:20]

x,y=map(list,zip(*top_unigrams))

sns.barplot(x=y,y=x, ax=axes[0,0], color='dodgerblue')





top_bigrams=get_top_bigrams(abstract)[:20]

x,y=map(list,zip(*top_bigrams))

sns.barplot(x=y,y=x, ax=axes[0,1], color='orangered')



top_threegrams=get_top_threegrams(abstract)[:20]

x,y=map(list,zip(*top_threegrams))

sns.barplot(x=y,y=x, ax=axes[1, 0], color='limegreen')



top_fourgrams=get_top_fourgrams(abstract)[:20]

x,y=map(list,zip(*top_fourgrams))

sns.barplot(x=y,y=x, ax=axes[1, 1], color='red')





axes[0, 0].set_ylabel(' ')

axes[0, 1].set_ylabel(' ')

axes[1, 0].set_ylabel(' ')

axes[1, 1].set_ylabel(' ')



axes[0, 0].yaxis.set_tick_params(labelsize=15)

axes[0, 1].yaxis.set_tick_params(labelsize=15)

axes[1, 0].yaxis.set_tick_params(labelsize=15)

axes[1, 1].yaxis.set_tick_params(labelsize=15)



axes[0, 0].set_title('Top 20 most common unigrams in abstract', fontsize=15)

axes[0, 1].set_title('Top 20 most common bigrams in abstract', fontsize=15)

axes[1, 0].set_title('Top 20 most common threegrams in abstract', fontsize=15)

axes[1, 1].set_title('Top 20 most common fourgrams in abstract', fontsize=15)



plt.tight_layout()

plt.show()
text = biorxiv['text'].apply(lambda x : clean_text(x))



plt.style.use('ggplot')

fig, axes = plt.subplots(2, 2, figsize=(18, 20), dpi=100)

plt.tight_layout()



top_unigrams=get_top_unigrams(text)[:20]

x,y=map(list,zip(*top_unigrams))

sns.barplot(x=y,y=x, ax=axes[0,0], color='dodgerblue')





top_bigrams=get_top_bigrams(text)[:20]

x,y=map(list,zip(*top_bigrams))

sns.barplot(x=y,y=x, ax=axes[0,1], color='orangered')



top_threegrams=get_top_threegrams(text)[:20]

x,y=map(list,zip(*top_threegrams))

sns.barplot(x=y,y=x, ax=axes[1, 0], color='limegreen')



top_fourgrams=get_top_fourgrams(text)[:20]

x,y=map(list,zip(*top_fourgrams))

sns.barplot(x=y,y=x, ax=axes[1, 1], color='red')





axes[0, 0].set_ylabel(' ')

axes[0, 1].set_ylabel(' ')

axes[1, 0].set_ylabel(' ')

axes[1, 1].set_ylabel(' ')



axes[0, 0].yaxis.set_tick_params(labelsize=15)

axes[0, 1].yaxis.set_tick_params(labelsize=15)

axes[1, 0].yaxis.set_tick_params(labelsize=15)

axes[1, 1].yaxis.set_tick_params(labelsize=15)



axes[0, 0].set_title('Top 20 most common unigrams in text', fontsize=15)

axes[0, 1].set_title('Top 20 most common bigrams in text', fontsize=15)

axes[1, 0].set_title('Top 20 most common threegrams in text', fontsize=15)

axes[1, 1].set_title('Top 20 most common fourgrams in text', fontsize=15)



plt.tight_layout()

plt.show()
plt.style.use('fivethirtyeight')

fig,(ax1,ax2, ax3)= plt.subplots(ncols=3, figsize=(18, 5), dpi=100)





length=title.str.split().map(lambda x: len(x))

ax1.hist(length,bins = 20, color='black')

ax1.set_title('Tittle')



length=abstract.str.split().map(lambda x: len(x))

ax2.hist(length, bins = 20,  color='black')

ax2.set_title('Abstract')



length=text.str.split().map(lambda x: len(x))

ax3.hist(length, bins = 20,  color='black')

ax3.set_title('Text')



plt.tight_layout()

plt.show()
# Covert the raw text into list

text = biorxiv.text.values.tolist()

print(text[:1])
# cleaning the text



text = [re.sub('\S*@\S*\s?', '', word) for word in text]

text = [re.sub('\s+', ' ', word) for word in text]

text = [re.sub("\'", "", word) for word in text]



print(text[:1])
# Tokenize words

def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



text_words = list(sent_to_words(text))



print(text_words[:1])


"""Build the bigram and trigram models"""

#bigram = gensim.models.Phrases(text_words, min_count=5, threshold=100) # higher threshold fewer phrases.

#trigram = gensim.models.Phrases(bigram[text_words], threshold=100)  



"""Faster way to get a sentence clubbed as a trigram/bigram"""

#bigram_mod = gensim.models.phrases.Phraser(bigram)

#trigram_mod = gensim.models.phrases.Phraser(trigram)



# See trigram example

#print(trigram_mod[bigram_mod[text_words[0]]])



"""Save an exported collocation model."""

#bigram_mod.save("/kaggle/working/my_bigram_model.pkl") 

#trigram_mod.save("/kaggle/working/my_trigram_model.pkl")

"""load an exported collocation model"""

bigram_reloaded = gensim.models.phrases.Phraser.load("../input/bi-and-tri-model/my_bigram_model.pkl")

trigram_reloaded = gensim.models.phrases.Phraser.load("../input/bi-and-tri-model/my_trigram_model.pkl")

print(trigram_reloaded[bigram_reloaded[text_words[0]]])
# NLTK Stop words

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

stop_words.extend(['et', 'al'])



# Define functions for stopwords, bigrams, trigrams and lemmatization

def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



def make_bigrams(texts):

    return [bigram_reloaded[doc] for doc in texts]



def make_trigrams(texts):

    return [trigram_reloaded[bigram_reloaded[doc]] for doc in texts]



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """https://spacy.io/api/annotation"""

    texts_out = []

    for word in texts:

        doc = nlp(" ".join(word)) 

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out
# Remove Stop Words

text_words_nostops = remove_stopwords(text_words)



# Form Bigrams

text_words_bigrams = make_bigrams(text_words_nostops)



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)

nlp = spacy.load('en', disable=['parser', 'ner'])



# Do lemmatization keeping only noun, adj, vb, adv

text_words_lemmatized = lemmatization(text_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])



print(text_words_lemmatized[:1])
# Create Dictionary

id2word = corpora.Dictionary(text_words_lemmatized)



# Create Corpus

texts = text_words_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]



# View

print(corpus[:1])
# Human readable format of corpus (term-frequency)

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
"""Build LDA model"""

#lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           #id2word=id2word,

                                           #num_topics=20, 

                                           #random_state=100,

                                           #update_every=1,

                                           #chunksize=100,

                                           #passes=10,

                                           #alpha='auto',

                                           #per_word_topics=True)

"""save model"""

#lda_model.save('/kaggle/working/lda_model.model')
# load trained model from file

model_reloaded =  gensim.models.LdaModel.load('../input/bi-and-tri-model/lda_model.model')



# Print the Keyword in the 10 topics

print(model_reloaded.print_topics())

doc_lda = model_reloaded[corpus]
"""Compute Perplexity"""

#print('\nPerplexity: ', model_reloaded.log_perplexity(corpus))  # a measure of how good the model is. lower the better.



"""Compute Coherence Score"""

#coherence_model_lda = CoherenceModel(model=model_reloaded, texts=text_words_lemmatized, dictionary=id2word, coherence='c_v')

#coherence_lda = coherence_model_lda.get_coherence()

#print('\nCoherence Score: ', coherence_lda)

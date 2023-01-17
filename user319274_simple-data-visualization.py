# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv',dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
#print(data.head(5))

import glob
all_json = glob.glob('../input/CORD-19-research-challenge/**/*.json', recursive=True)
len(all_json)
import json
with open(all_json[0]) as file:
    article1 = json.load(file)
    #print(json.dumps(article1, indent = 4))
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.title = content['metadata']['title']
            self.abstract = []
            self.body_text = []
            # Abstract
            if content.get("abstract") != None:
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
            # Extend Here
            #
            #
    def __repr__(self):
        return f'{self.paper_id}: {self.title}: {self.abstract[:200]}... {self.body_text[:200]}...'
 
first_row = FileReader(all_json[0])
#print(first_row)
dict_ = {'paper_id': [], 'title': [], 'abstract': [], 'body_text': []}
for ind, entry in enumerate(all_json):
    if ind % (len(all_json) // 10) == 0:
        print(f'Processing index: {ind} of {len(all_json)}')
    content = FileReader(entry)
    dict_['paper_id'].append(content.paper_id)
    dict_['title'].append(content.title)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'title', 'abstract', 'body_text'])
df_covid.head()
import re

df_title = df_covid.loc[:, ["title"]].dropna()
df_title["title"] = df_title['title'].apply(lambda x: x.lower())
df_title["title"] = df_title['title'].apply(lambda x: x.strip())
df_title["title"] = df_title['title'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
df_title["title"] = df_title['title'].apply(lambda x: re.sub(' +',' ',x))
titles = ' '.join(df_title["title"])

#print(titles[:100])
from collections import Counter
import matplotlib.pyplot as plt

titles_word_list = titles.split()

titles_word_count = Counter(titles_word_list).most_common()

new_titles_word_count = [i for i in titles_word_count if i[0] not in ['of', 'and', 'in', 'the', 'a', 
                                                                      'for', 'with', 'to', 'from', 'by', 'on',
                                                                     'an', 'at', 'as', 'is', 'are']]

labels, values = zip(*new_titles_word_count)

indexes = np.arange(len(labels[:100]))
width = 1

plt.figure(num=None, figsize=(16, 14), dpi=80, facecolor='w', edgecolor='k')

plt.bar(indexes, values[:100], width)
plt.xticks(indexes + width * 0.5, labels[:100], fontsize=10, rotation='vertical')
plt.show()

import matplotlib
import squarify

cmap = matplotlib.cm.Blues
norm = matplotlib.colors.Normalize(vmin=min(values[:100]), vmax=max(values[:100]))
colors = [cmap(norm(value)) for value in values[:100]]

plt.figure(num=None, figsize=(16, 14), dpi=80, facecolor='w', edgecolor='k')

squarify.plot(label=labels[:100],sizes=values[:100], alpha=.8, color = colors)
plt.title("Title Word Count",fontsize=18,fontweight="bold")

plt.axis('off')
plt.show()
dict2_ = {'paper_id': [], 'title': [], 'abstract': [], 'body_text': [], 'publish_time': []}
for ind, entry in enumerate(all_json):
    if ind % (len(all_json) // 10) == 0:
        print(f'Processing index: {ind} of {len(all_json)}')
    content = FileReader(entry)
    
    # get metadata information
    meta_data = data.loc[data['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue
    
    
    dict2_['paper_id'].append(content.paper_id)
    dict2_['title'].append(content.title)
    dict2_['abstract'].append(content.abstract)
    dict2_['body_text'].append(content.body_text)
    dict2_['publish_time'].append(meta_data['publish_time'].values[0])
df_covid2 = pd.DataFrame(dict2_, columns=['paper_id', 'title', 'abstract', 'body_text', 'publish_time'])
df_covid2.head()
df_covid2_subset = df_covid2.dropna()
df_covid2_subset['publish_year'] = df_covid2_subset['publish_time'].apply(lambda x: x[:4])
df_covid2_subset.head()
df_covid2_subset['title_word_count'] = df_covid2_subset['title'].apply(lambda x: len(x.strip().split()))
df_covid2_subset['abstract_word_count'] = df_covid2_subset['abstract'].apply(lambda x: len(x.strip().split()))
df_covid2_subset['body_word_count'] = df_covid2_subset['body_text'].apply(lambda x: len(x.strip().split()))
df_covid2_subset.head()
df_covid2_subset['title_COVID-19_count'] = df_covid2_subset['title'].apply(lambda x: len([i for i, e in enumerate(x.strip().split()) if e == 'coronavirus']))
df_covid2_subset['abstract_COVID-19_count'] = df_covid2_subset['abstract'].apply(lambda x: len([i for i, e in enumerate(x.strip().split()) if e == 'coronavirus']))
df_covid2_subset['body_COVID-19_count'] = df_covid2_subset['body_text'].apply(lambda x: len([i for i, e in enumerate(x.strip().split()) if e == 'coronavirus']))
df_covid2_subset.head()
word_sum_pvt = pd.pivot_table(df_covid2_subset, values=['title_COVID-19_count', 'abstract_COVID-19_count', 'body_COVID-19_count'], 
                              index=['publish_year'], aggfunc=np.sum, fill_value=0)


word_sum_pvt.plot(figsize=(16,14))

#from sklearn.feature_extraction.text import CountVectorizer

#count_vectorizer = CountVectorizer(stop_words='english')
#count_data = count_vectorizer.fit_transform(df_covid2_subset['body_text'][:50])

import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(df_covid2_subset['body_text'][:100]))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Define functions for stopwords, bigrams, trigrams and lemmatization
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

print(lda_model.print_topics())
doc_lda = lda_model[corpus]
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
%matplotlib inline

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis
from sklearn.decomposition import LatentDirichletAllocation as LDA

number_topics = 5
number_words = 10

lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_ind, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_ind)
        print(" ".join([words[i]
            for i in topic.argsort()[:-n_top_words -1:-1]]))

print_topics(lda, count_vectorizer, number_words)
from pyLDAvis import sklearn as sklearn_lda
import pyLDAvis
import pyLDAvis.gensim

LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
pyLDAvis.display(LDAvis_prepared)
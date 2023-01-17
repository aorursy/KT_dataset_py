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
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import json
import os
import string
from os import listdir
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation as LDA
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis
import gensim
# Run this code block only once.
comm_files = listdir('/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json')
noncomm_files = listdir('/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json')
biorxiv_medrxiv_files = listdir('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json')
custom_license_files = listdir('/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json')
comm_basePath = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json'
noncomm_basePath = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json'
biomed_basePath = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json'
custom_basePath = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json'

print(len(comm_files))
print(len(noncomm_files))
print(len(biorxiv_medrxiv_files))
print(len(custom_license_files))

def preprocessJSON(json_infile, basepath):
    json_input = json.load(open(basepath + '/' + json_infile))
    json_body = json_normalize(data=json_input,
                                 record_path='body_text',
                                 meta=[['metadata','title']])
    text_array = json_body.iloc[:,0].values
    text_df = pd.DataFrame({'main text':[' '.join(text_array)]},
                           index=[json_body.iloc[0,4]])
    return text_df

all_comm_main_text = [preprocessJSON(f, comm_basePath) for f in comm_files]
comm_main_df = pd.concat(all_comm_main_text)
comm_main_df.to_csv('comm_used_main_df.csv')

all_noncomm_main_text = [preprocessJSON(f, noncomm_basePath) for f in noncomm_files]
noncomm_main_df = pd.concat(all_noncomm_main_text)
noncomm_main_df.to_csv('noncomm_used_main_df.csv')

all_biomed_main_text = [preprocessJSON(f, biomed_basePath) for f in biorxiv_medrxiv_files]
biomed_main_df = pd.concat(all_biomed_main_text)
biomed_main_df.to_csv('biomed_main_df.csv')

all_custom_main_text = [preprocessJSON(f, custom_basePath) for f in custom_license_files]
custom_main_df = pd.concat(all_custom_main_text)
custom_main_df.to_csv('custom_main_df.csv')
# Read in pre-processed csv files containing each paper's title and the corresponding main text.
wkdir = '/kaggle/working/'
comm_used_corpus = pd.read_csv(wkdir + 'comm_used_main_df.csv', index_col=0)
noncomm_used_corpus = pd.read_csv(wkdir + 'noncomm_used_main_df.csv', index_col=0)
biomed_corpus = pd.read_csv(wkdir + 'biomed_main_df.csv', index_col=0)
custom_corpus = pd.read_csv(wkdir + 'custom_main_df.csv', index_col=0)
all_corpus = pd.concat([comm_used_corpus,
                        noncomm_used_corpus,
                        biomed_corpus,
                        custom_corpus])

all_corpus.isnull().values.any()
title = all_corpus.index.to_list()
title = [str(i) if title[i] != title[i] else title[i] for i in np.arange(len(title))]
all_corpus.index = title
print(all_corpus.shape)
# Helper function
def regex_func(text):
    if text:
        res = re.search('COVID | coronavirus | Coronavirus | SARS-Cov | SARS-nCov',text)
        return res
    else:
        return False

# Here we use a Porter stemmer to stem suffixes from words 
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
regex_expr = '\\b(?:COVID|coronavirus|Coronavirus|SARS-Cov|SARS-nCov|sars|corona)'
all_corpus = all_corpus[all_corpus.index.str.contains(regex_expr) & all_corpus.loc[:,'main text'].str.contains(regex_expr)]
all_corpus
def processText(text):
    lower_text = text.lower()
    nonpunc_text = lower_text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens,stemmer)
    # Remove words containing non-word characters
    return [word for word in tokens if re.match(r'[^\W\d]*$', word) and len(word) >= 2]

corpus_list = [processText(all_corpus.iloc[i,0]) for i in range(int(all_corpus.count()))]

fn = 'COVID_corpus_list.txt'
### Run this code block only once and comment out the ###
### line generating corpus_list afterwards ##############
f = open(fn,'w')
for l in corpus_list:
    f.write(' '.join(l) + '\n')
f.close()
#########################################################
f = open(fn,'r')
corpus_list = f.readlines()
corpus_list = [l.strip().split() for l in corpus_list]
f.close()
corpus_dict = gensim.corpora.Dictionary(corpus_list)
corpus_dict.filter_extremes(no_below=15, no_above=0.55, keep_n=100000)
corpus_bow = [corpus_dict.doc2bow(doc) for doc in corpus_list]
corpus_lda = gensim.models.LdaMulticore(corpus_bow, 
                                        num_topics=20, 
                                        id2word=corpus_dict,
                                        passes=4,
                                        workers=1)
for idx, topic in corpus_lda.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
print('\nPerplexity: ', corpus_lda.log_perplexity(corpus_bow))

# Compute Coherence Score
from gensim.models.coherencemodel import CoherenceModel
coherence_model_lda = CoherenceModel(model=corpus_lda, 
                                     texts=corpus_list, 
                                     dictionary=corpus_dict, 
                                     coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
pyLDAvis.enable_notebook()
import pyLDAvis.gensim
vis = pyLDAvis.gensim.prepare(corpus_lda, corpus_bow, corpus_dict)
vis
corpus_dict = {}
for i in range(int(all_corpus.count())):
    curr_text = all_corpus.iloc[i,0]
    lower_text = curr_text.lower()
    nonpunc_text = lower_text.translate(str.maketrans('', '', string.punctuation))
    # Remove words containing non-word characters and short words
    split_text = nonpunc_text.split()
    cleaned_text = ' '.join([word for word in split_text if re.match(r'[^\W\d]*$', word) and len(word) >= 2])
    corpus_dict[title[i]] = cleaned_text
    
def tokenizer(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
    
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english')
tfs = tfidf.fit_transform(corpus_dict.values())

# Read in the queries
in_fn = '/kaggle/input/searchterms/searchTerms.txt'
f = open(in_fn,'r')
queries = f.readlines()
f.close()
queries = [query.strip() for query in queries]

query_dict = {}
for i in range(len(queries)):
    curr_text = queries[i]
    lower_text = curr_text.lower()
    nonpunc_text = lower_text.translate(str.maketrans('', '', string.punctuation))
    # Remove words containing non-word characters and short words
    split_text = nonpunc_text.split()
    cleaned_text = ' '.join([word for word in split_text if re.match(r'[^\W\d]*$', word) and len(word) >= 2])
    query_dict[i] = cleaned_text

query_tfs = tfidf.transform(query_dict.values())
from sklearn.metrics.pairwise import cosine_similarity
cos_sim = [cosine_similarity(qtf, tfs).flatten() for qtf in query_tfs]
top_article_indices = np.argsort(cos_sim,axis=1)
top_article_indices = [np.flip(tai) for tai in top_article_indices]
count = 0
for idx in top_article_indices:
    print(queries[count].strip(".")+":\n")
    for i in range(5):
        print(str(i+1) + '. ' + all_corpus.index[idx[i]] + '\n')
    print('===============================================================================\n')
    count += 1

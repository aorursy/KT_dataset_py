import numpy as np

import pandas as pd

import seaborn as sns



import string

import collections

 

from nltk import word_tokenize

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import silhouette_score

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.manifold import MDS



import sklearn

from sklearn.semi_supervised import LabelPropagation

from sklearn.semi_supervised import LabelSpreading

from sklearn.metrics import classification_report

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder



from sklearn.metrics import confusion_matrix



import scipy as sc

from scipy.cluster.hierarchy import fcluster

from scipy.cluster.hierarchy import ward, dendrogram



import gensim

from gensim import corpora

from gensim.models.word2vec import Word2Vec

from gensim.models import KeyedVectors



# libraries for visualization

import pyLDAvis

import pyLDAvis.gensim

import matplotlib.pyplot as plt



from itertools import chain



import json



%matplotlib inline
df = pd.read_csv('../input/tales_data_set.csv')
df
df_train = df.copy()
for i in range(len(df_train)):

    # leave approximately 80% for training

    if np.random.random() > 0.8:

        df_train.loc[i, 'Label'] = None
punctuation = string.punctuation



def process_text(text):

    """Remove punctuation, lower text, tokenize text and stem tokens (words).

    Args:

      text: a string.

    Returns:

      Tokenized text i.e. list of stemmed words. 

    """

    

    # replace all punctuation by blanc spaces

    trantab = str.maketrans(punctuation, ' '*len(punctuation))

    text = text.translate(trantab)

    

    # lower text

    text = text.lower()

    

    # tokenize text

    tokens = word_tokenize(text) 

  

    # remove stop words

    # filtered_text = [w for w in word_tokens if not w in stopwords] 

        

    # stemm text

    # stemmer = PorterStemmer()

    # tokens = [stemmer.stem(t) for t in filtered_text]



    return tokens
def define_vectorizer(add_stop_words = [], max_df=0.5, min_df=0.1, ngram_range=(1, 1)):

    """Transform texts to Tf-Idf coordinates.

    Args:

      add_stop_words: addititional stop_words, list of strings.

      ngram_range: tuple (min_n, max_n) (default=(1, 1))

        The lower and upper boundary of the range of n-values for different n-grams to be extracted.

        All values of n such that min_n <= n <= max_n will be used.

      max_df: float in range [0.0, 1.0] or int

        When building the vocabulary ignore terms that have a document frequency strictly higher than

        the given threshold (corpus-specific stop words).

        If float, the parameter represents a proportion of documents, integer absolute counts.

        This parameter is ignored if vocabulary is not None.

      min_df: float in range [0.0, 1.0] or int

        When building the vocabulary ignore terms that have a document frequency strictly lower than

        the given threshold. This value is also called cut-off in the literature.

        If float, the parameter represents a proportion of documents, integer absolute counts.

        This parameter is ignored if vocabulary is not None.

    Returns:

      Vectorizer. 

    """

    vectorizer = TfidfVectorizer(tokenizer=process_text,

                             #stop_words=stopwords.words('english') + add_stop_words,

                             max_df=max_df,

                             min_df=min_df,

                             ngram_range=ngram_range)#,

                             #lowercase=True)

    return vectorizer



def compute_tfidf_matrix(corpus, vectorizer):

    """Transform texts to Tf-Idf coordinates.

    Args:

      corpus: list of strings.

      vectorizer: sklearn TfidfVectorizer.

    Returns:

      A sparse matrix in Compressed Sparse Row format with tf-idf scores 

    Raises:

      ValueError: If `corpus` generates empty vocabulary or after pruning no terms remain.

    """

    tfidf_matrix = vectorizer.fit_transform(corpus) # raises ValueError if bad corpus

    return tfidf_matrix



def corpus_features(df, field):

    corpus = df[field].values

    vectorizer = define_vectorizer()

    tfidf_matrix = compute_tfidf_matrix(corpus = corpus, vectorizer = vectorizer)

    dist = 1 - cosine_similarity(tfidf_matrix)

    return vectorizer, tfidf_matrix, dist
def auto_fill_in(df, field = 'Label'):

    """

    df - data frame

    """

    assert field in df.columns, field+' is not in df columns'

    df_ri = df.reset_index(drop=True)

    train_ind = df_ri.loc[~df_ri[field].isna()].index

    target_ind = df_ri.loc[df_ri[field].isna()].index

    assert len(train_ind) > 0, 'There is no labeled data in df'

    assert len(target_ind) > 0, 'There is no unlabeled data in df'

    train = df_ri.iloc[train_ind]

    target = df_ri.iloc[target_ind]

    

    texts = df_ri['Tale'].apply(lambda x: x.lower().replace('\n', ''))

    vectorizer = define_vectorizer()

    X = compute_tfidf_matrix(texts.values, vectorizer)

    

    X_train = X[train_ind]

    X_target = X[target_ind]

    

    #y_train = train['cluster'].values

    #y_train = y_train.astype('int')

    le = LabelEncoder()

    y_train = le.fit_transform(train[field].values)

        

    KNN = KNeighborsClassifier(n_neighbors=7) # any n_neighbors is fine

    KNN.fit(X_train, y_train) 

    

    y_target = KNN.predict(X_target)

    y_cluster = le.inverse_transform(y_target)

    

    df_ri.loc[target_ind, field] = y_cluster

    

    return df_ri
df_auto_fill_in = auto_fill_in(df_train)
confusion_matrix(df['Label'], df_auto_fill_in['Label'])
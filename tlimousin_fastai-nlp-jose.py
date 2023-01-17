# Esto viene de

# https://github.com/fastai/course-nlp/blob/master/2-svd-nmf-topic-modeling.ipynb

# https://www.fast.ai/2019/07/08/fastai-nlp/
import numpy as np

from sklearn.datasets import fetch_20newsgroups

from sklearn import decomposition

from scipy import linalg

import matplotlib.pyplot as plt
%matplotlib inline

np.set_printoptions(suppress=True)
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

remove = ('headers', 'footers', 'quotes')

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)

newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)
newsgroups_train.filenames.shape, newsgroups_train.target.shape
print("\n".join(newsgroups_train.data[:3]))
np.array(newsgroups_train.target_names)[newsgroups_train.target[:3]]
newsgroups_train.target[:10]
num_topics, num_top_words = 6, 8
from sklearn.feature_extraction import stop_words



sorted(list(stop_words.ENGLISH_STOP_WORDS))[:20]
import nltk

nltk.download('wordnet')
from nltk import stem
wnl = stem.WordNetLemmatizer()

porter = stem.porter.PorterStemmer()
word_list = ['feet', 'foot', 'foots', 'footing']
[wnl.lemmatize(word) for word in word_list]
[porter.stem(word) for word in word_list]
import spacy
from spacy.lemmatizer import Lemmatizer

lemmatizer = Lemmatizer()
[lemmatizer.lookup(word) for word in word_list]
nlp = spacy.load("en_core_web_sm")
sorted(list(nlp.Defaults.stop_words))[:20]
#Exercise:

#Exercise:

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk

# nltk.download('punkt')
# from nltk import word_tokenize



# class LemmaTokenizer(object):

#     def __init__(self):

#         self.wnl = stem.WordNetLemmatizer()

#     def __call__(self, doc):

#         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
vectorizer = CountVectorizer(stop_words='english') #, tokenizer=LemmaTokenizer())
vectors = vectorizer.fit_transform(newsgroups_train.data).todense() # (documents, vocab)

vectors.shape #, vectors.nnz / vectors.shape[0], row_means.shape
print(len(newsgroups_train.data), vectors.shape)
vocab = np.array(vectorizer.get_feature_names())
vocab.shape
vocab[7000:7020]
%time U, s, Vh = linalg.svd(vectors, full_matrices=False)
print(U.shape, s.shape, Vh.shape)
s[:4]
np.diag(np.diag(s[:4]))
#Exercise: confrim that U, s, Vh is a decomposition of `vectors`

#Exercise: Confirm that U, Vh are orthonormal

plt.plot(s);
plt.plot(s[:10])
num_top_words=8



def show_topics(a):

    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]

    topic_words = ([top_words(t) for t in a])

    return [' '.join(t) for t in topic_words]
show_topics(Vh[:10])
m,n=vectors.shape

d=5  # num topics
clf = decomposition.NMF(n_components=d, random_state=1)



W1 = clf.fit_transform(vectors)

H1 = clf.components_
show_topics(H1)
vectorizer_tfidf = TfidfVectorizer(stop_words='english')

vectors_tfidf = vectorizer_tfidf.fit_transform(newsgroups_train.data) # (documents, vocab)
newsgroups_train.data[10:20]
W1 = clf.fit_transform(vectors_tfidf)

H1 = clf.components_
show_topics(H1)
plt.plot(clf.components_[0])
clf.reconstruction_err_
%time u, s, v = np.linalg.svd(vectors, full_matrices=False)
from sklearn import decomposition

import fbpca
%time u, s, v = decomposition.randomized_svd(vectors, 10)
%time u, s, v = fbpca.pca(vectors, 10)
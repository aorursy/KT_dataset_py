# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os.path

import sys

import re

import itertools

import csv

import datetime

import pickle

from collections import defaultdict, Counter

import random

import gc



import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import seaborn as sns

import pandas as pd

import numpy as np

import scipy

import gensim

from sklearn.metrics import f1_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
'''

that's copy of keras load_data function

'''

from __future__ import absolute_import



from six.moves import zip

import numpy as np

import json

import warnings





def load_data(path='reuters.npz', num_words=None, skip_top=0,

              maxlen=None, test_split=0.2, seed=113,

              start_char=1, oov_char=2, index_from=3, **kwargs):

    """Loads the Reuters newswire classification dataset.



    # Arguments

        path: where to cache the data (relative to `~/.keras/dataset`).

        num_words: max number of words to include. Words are ranked

            by how often they occur (in the training set) and only

            the most frequent words are kept

        skip_top: skip the top N most frequently occuring words

            (which may not be informative).

        maxlen: truncate sequences after this length.

        test_split: Fraction of the dataset to be used as test data.

        seed: random seed for sample shuffling.

        start_char: The start of a sequence will be marked with this character.

            Set to 1 because 0 is usually the padding character.

        oov_char: words that were cut out because of the `num_words`

            or `skip_top` limit will be replaced with this character.

        index_from: index actual words with this index and higher.



    # Returns

        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.



    Note that the 'out of vocabulary' character is only used for

    words that were present in the training set but are not included

    because they're not making the `num_words` cut here.

    Words that were not seen in the training set but are in the test set

    have simply been skipped.

    """

    # Legacy support

    if 'nb_words' in kwargs:

        warnings.warn('The `nb_words` argument in `load_data` '

                      'has been renamed `num_words`.')

        num_words = kwargs.pop('nb_words')

    if kwargs:

        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))



    path = path

    npzfile = np.load(path)

    xs = npzfile['x']

    labels = npzfile['y']

    npzfile.close()



    np.random.seed(seed)

    np.random.shuffle(xs)

    np.random.seed(seed)

    np.random.shuffle(labels)



    if start_char is not None:

        xs = [[start_char] + [w + index_from for w in x] for x in xs]

    elif index_from:

        xs = [[w + index_from for w in x] for x in xs]



    if maxlen:

        new_xs = []

        new_labels = []

        for x, y in zip(xs, labels):

            if len(x) < maxlen:

                new_xs.append(x)

                new_labels.append(y)

        xs = new_xs

        labels = new_labels



    if not num_words:

        num_words = max([max(x) for x in xs])



    # by convention, use 2 as OOV word

    # reserve 'index_from' (=3 by default) characters:

    # 0 (padding), 1 (start), 2 (OOV)

    if oov_char is not None:

        xs = [[oov_char if (w >= num_words or w < skip_top) else w for w in x] for x in xs]

    else:

        new_xs = []

        for x in xs:

            nx = []

            for w in x:

                if skip_top <= w < num_words:

                    nx.append(w)

            new_xs.append(nx)

        xs = new_xs



    x_train = np.array(xs[:int(len(xs) * (1 - test_split))])

    y_train = np.array(labels[:int(len(xs) * (1 - test_split))])



    x_test = np.array(xs[int(len(xs) * (1 - test_split)):])

    y_test = np.array(labels[int(len(xs) * (1 - test_split)):])



    return (x_train, y_train), (x_test, y_test)



def get_word_index(path='reuters_word_index.json'):

    """Retrieves the dictionary mapping word indices back to words.



    # Arguments

        path: where to cache the data (relative to `~/.keras/dataset`).



    # Returns

        The word index dictionary.

    """

    path = path

    f = open(path)

    data = json.load(f)

    f.close()

    return data
print('Loading data...')

path = '../input/reuters.npz'

(docs, _), (_, _) = load_data(path=path, test_split=0.0, start_char=None)

print(len(docs), 'sequences')
word_index = get_word_index(path="../input/reuters_word_index.npz")

word_index['<oov>'] = 0

print(len(word_index))
word_dic = gensim.corpora.Dictionary(prune_at=None)

word_dic.token2id = word_index



print('[the] >>>', word_dic.token2id['the'])

print('[a] >>>', word_dic.token2id['a'])

print('[in] >>>', word_dic.token2id['in'])

print('[1986] >>>', word_dic.token2id['1986'])

print('[finance] >>>', word_dic.token2id['finance'])

print('')

print('id 0 >>>', word_dic[0])

print('id 1 >>>', word_dic[1])

print('id 27595 >>>', word_dic[27595])

print('id 30979 >>>', word_dic[30979])

print('')

print('len(word_dic) >>>', len(word_dic))

print('len(word_dic.token2id) >>>', len(word_dic.token2id))
' '.join([word_dic[ee-3] for ee in docs[0]])
'''

sentences for gensim

'''

class Sentences(object):

    def __init__(self, docs, word_dic):

        self.docs = docs

        self.word_dic = word_dic

    

    def __len__(self):

        return self.docs.shape[0]

    

    def __iter__(self):

        for irow in self.docs:

            yield [self.word_dic[ee-3] if 3<ee else '<oov>' for ee in irow]
sentences = Sentences(docs, word_dic)

len(sentences)
for ee in itertools.islice(sentences, None):

    if 'northerly' in ee:

        print(ee)

        print(word_dic.doc2bow(ee))

        break

'''

"northerly" does not exist, delete from dictionary

'''

print(len(word_dic))

del word_dic.token2id['northerly']

del word_dic.id2token[30979]

len(word_dic)
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)



'''

make tfidf

'''

sentences = Sentences(docs, word_dic)

len(sentences)



tfidf = gensim.models.TfidfModel((word_dic.doc2bow(ee) for ee in itertools.islice(sentences, None)), id2word=word_dic)

#tfidf.dfs
class MySparseMatrixSimilarity(gensim.similarities.docsim.SparseMatrixSimilarity):

    DTYPE = np.float32

    

    def __init__(self, corpus_csr, tfidf, num_features=None, num_terms=None, num_docs=None, num_nnz=None,

                 num_best=None, chunksize=500, dtype=np.float32, maintain_sparsity=False):

        super(MySparseMatrixSimilarity, self).__init__(None, num_features=num_features, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,

                 num_best=num_best, chunksize=chunksize, dtype=dtype, maintain_sparsity=maintain_sparsity)

        self.index = corpus_csr

        self.index_csc = corpus_csr.tocsc()

        self.tfidf = tfidf

        self.normalize = False

        self.method = None

        self.SLOPE = 0.2

        

        idfs = np.empty((max(tfidf.idfs.keys())+1,1), dtype=self.DTYPE)

        for termid in tfidf.idfs:

            v = tfidf.idfs[termid]

            idfs[termid,0] = v

        self.idfs = idfs

        

        self.getDF()

    

    def getDF(self):

        ll = []

        for ii in range(self.index.shape[0]):

            ll.append(self.index.getrow(ii).nnz)

        self.DF = np.array(ll, dtype=self.DTYPE)

        

    def get_nonzero_idx2(self, ix):

        tmp = self.index_csc[:,ix]

        nonzero_idx2 = list(set(tmp.nonzero()[0]))

        nonzero_idx2.sort()

        return nonzero_idx2

    

    def _calc(self, naiseki, norm, nonzero_idx2):

        res = naiseki.multiply(norm.reshape((norm.shape[0],1)))

        res2 = scipy.sparse.csr_matrix((self.index.shape[0], 1), dtype=self.DTYPE)

        res2[nonzero_idx2,:] = res

        return res2

    

    def getCorpusByDoc(self, docid, method='WT_SMART'):

        query = self.index[docid]

        if method in ['WT_TF', 'WT_TFIDF', 'WT_SMART', 'WT_SMART2']:

            return self.calc_wq(query, method=method)

        else:

            raise Exception('no such method [%s]' % method)

    

    def calc_wq(self, query, method='WT_SMART'):

        if query.count_nonzero() == 0:

            raise ValueError('freq must be more than zero')

        if method == 'WT_SMART':

            wq = self.calc_tfn_mx(query)

            wq = wq.T.multiply(self.idfs) # tf_n(t|q) * idf(t)

            return wq.T

        elif method == 'WT_SMART2':

            wq = self.calc_tfn_mx2(query)

            wq = wq.T.multiply(self.idfs) # tf_n(t|q) * idf(t)

            return wq.T

        elif method == 'WT_TF':

            wq = query.tocsc()

            return wq

        elif method == 'WT_TFIDF':

            wq = query.tocsc()

            wq = wq.T.multiply(self.idfs)

            return wq.T

        else:

            raise Exception('no such method [%s]' % method)

    

    def calc_wd(self, mx, method='WT_SMART'):

        if method == 'WT_SMART':

            wd = self.calc_tfn_mx(mx)

            return wd

        elif method == 'WT_SMART2':

            wd = self.calc_tfn_mx2(mx)

            return wd

        elif method in ['WT_TF', 'WT_TFIDF']:

            return mx

        else:

            raise Exception('no such method [%s]' % method)

    

    def calc_norm(self, wq, wd, nonzero_idx2, method='WT_SMART'):

        if method == 'WT_SMART':

            norm = 1 / ((1-self.SLOPE)*self.DF.mean() + self.SLOPE*self.DF[nonzero_idx2])

            return norm

        elif method == 'WT_SMART2':

            norm = 1 / ((1-self.SLOPE)*self.DF.mean() + self.SLOPE*self.DF[nonzero_idx2])

            return norm

        elif method in ['WT_TF', 'WT_TFIDF']:

            ll = []

            for ii in range(wd.shape[0]):

                ll.append(wd.getrow(ii).sum())

            norm = np.array(ll, dtype=self.DTYPE)

            norm = 1 / norm

            return norm

        else:

            raise Exception('no such method [%s]' % method)

    

    def calc_tfn_mx(self, mx):

        '''

                    1 + log(TF(t|q))

        tf_n(t|q) = ---------------------

                    1 + log(ave(TF(.|q)))

        '''

        sums = mx.sum(axis=1)

        sums = np.array(sums.reshape(-1).tolist()[0])

        nnz = []

        for ii in range(mx.shape[0]):

            nnz.append(mx.getrow(ii).count_nonzero())

        nnz = np.array(nnz)

        means = 1 + np.log(sums / nnz)

        nonzero_idx = mx.nonzero()

        mx[nonzero_idx] = mx[nonzero_idx] - 1

        mx = mx.log1p()

        mx[nonzero_idx] = (1 + mx[nonzero_idx])

        vs = []

        for ii in range(mx.shape[0]):

            vs.append(mx[ii,].multiply(1 / means[ii]))

        mx2 = scipy.sparse.vstack(vs)

        return mx2

    

    def calc_tfn_mx2(self, mx):

        '''

                    log(1 + TF(t|q))

        tf_n(t|q) = ---------------------

                    ave(log(1 + TF(.|q)))

        '''

        nnz = []

        for ii in range(mx.shape[0]):

            nnz.append(mx.getrow(ii).count_nonzero())

        nnz = np.array(nnz)

        

        mx = mx.log1p()

        sums = mx.sum(axis=1)

        sums = np.array(sums.reshape(-1).tolist()[0])

        means = sums / nnz

        

        vs = []

        for ii in range(mx.shape[0]):

            vs.append(mx[ii,].multiply(1 / means[ii]))

        mx2 = scipy.sparse.vstack(vs)

        return mx2

    

    def calc_sim_WT_MINE(self, query):

        '''wq'''

        wq = query.tocsc().multiply(self.idfs)

        '''wd'''

        wd = self.index

        '''inner product'''

        naiseki = wd * wq

        '''norm'''

        ones = np.ones(self.index.shape[1])

        norm = 1 / self.index.dot(ones)

        res = naiseki.multiply(norm.reshape((norm.shape[0],1)))

        return res

    

    def calc_sim_WT_TF(self, query):

        '''

        WQ

        wq(t|q) = TF(t|q)

        '''

        #wq = query.tocsc()

        wq = self.calc_wq(query.T, method='WT_TF').T

        nonzero_idx = wq.nonzero()

        

        '''

        WD

        wd(t|d) = TF(t|d)

        '''

        nonzero_idx2 = self.get_nonzero_idx2(nonzero_idx[0])

        wd = self.index[nonzero_idx2,:].copy()

        

        '''inner product'''

        naiseki = wd.dot(wq)

        

        '''

        norm

        norm(d) = TF(.|d)

        '''

        ll = []

        for ii in range(wd.shape[0]):

            ll.append(wd.getrow(ii).sum())

        norm = np.array(ll, dtype=self.DTYPE)

        norm = 1 / norm

        

        return self._calc(naiseki, norm, nonzero_idx2)

    

    def calc_sim_WT_TFIDF(self, query):

        '''

        WQ

        wq(t|q) = TF(t|q) * idf(t)

        '''

        wq = self.calc_wq(query.T, method='WT_TFIDF').T

        nonzero_idx = wq.nonzero()

        

        '''

        WD

        wd(t|d) = TF(t|d)

        '''

        nonzero_idx2 = self.get_nonzero_idx2(nonzero_idx[0])

        wd = self.index[nonzero_idx2,:].copy()

        

        '''inner product'''

        naiseki = wd.dot(wq)

        

        '''

        norm

        norm(d) = TF(.|d)

        '''

        ll = []

        for ii in range(wd.shape[0]):

            ll.append(wd.getrow(ii).sum())

        norm = np.array(ll, dtype=self.DTYPE)

        norm = 1 / norm

        

        return self._calc(naiseki, norm, nonzero_idx2)

    

    def calc_sim_WT_SMART(self, query):

        '''

        sim(d|q) = 1 / norm(d) * \sum_t { wq(t|q) * wd(t|d) }

        wq(t|q) = tf_n(t|q) * idf(t)

        wd(t|d) = tf_n(t|d)

        '''

        '''

        WQ

                    1 + log(TF(t|q))

        tf_n(t|q) = ---------------------

                    1 + log(ave(TF(.|q)))

        '''

        wq = self.calc_wq(query.T).T

        nonzero_idx = wq.nonzero()

        

        '''

        WD

        tf_n(t|d) = 1 + log(TF(t|d))

        '''

        nonzero_idx2 = self.get_nonzero_idx2(nonzero_idx[0])

        wd = self.index[nonzero_idx2,].astype(self.DTYPE)

        wd = self.calc_tfn_mx(wd)

        

        '''inner product'''

        naiseki = wd.dot(wq)

        

        '''

        norm

        norm(d) = ave(len(.)) + slope * (len(d) - ave(len(.)))

        '''

        norm = 1 / ((1-self.SLOPE)*self.DF.mean() + self.SLOPE*self.DF[nonzero_idx2])

        

        ret = self._calc(naiseki, norm, nonzero_idx2)

        return ret

    

    def calc_sim_WT(self, query, method='WT_SMART'):

        '''

        sim(d|q) = 1 / norm(d) * \sum_t { wq(t|q) * wd(t|d) }

        '''

        '''

        wq

        '''

        wq = self.calc_wq(query.T, method=method).T

        nonzero_idx = wq.nonzero()

        

        '''

        wd

        '''

        nonzero_idx2 = self.get_nonzero_idx2(nonzero_idx[0])

        wd = self.index[nonzero_idx2,].astype(self.DTYPE)

        wd = self.calc_wd(wd, method=method)

        

        '''inner product'''

        naiseki = wd.dot(wq)

        

        '''

        norm

        norm(d) = ave(len(.)) + slope * (len(d) - ave(len(.)))

        '''

        norm = self.calc_norm(wq, wd, nonzero_idx2, method=method)

        

        ret = self._calc(naiseki, norm, nonzero_idx2)

        return ret

    

    def calc_sim(self, query):

        if self.method is None:

            ret = self.calc_sim_WT(query, method='WT_SMART')

            return ret

        elif self.method in ['WT_TFIDF', 'WT_TF', 'WT_SMART', 'WT_SMART2']:

            ret = self.calc_sim_WT(query, method=self.method)

            return ret

        elif self.method == 'WT_MINE':

            return self.calc_sim_WT_MINE(query)

        else:

            raise Exception('no such method [%s]' % self.method)

    

    def get_similarities(self, query):

        is_corpus, query = gensim.utils.is_corpus(query)

        if is_corpus:

            query = gensim.matutils.corpus2csc(query, self.index.shape[1], dtype=self.index.dtype)

        else:

            if scipy.sparse.issparse(query):

                query = query.T  # convert documents=rows to documents=columns

            elif isinstance(query, np.ndarray):

                if query.ndim == 1:

                    query.shape = (1, len(query))

                query = scipy.sparse.csr_matrix(query, dtype=self.index.dtype).T

            else:

                # default case: query is a single vector, in sparse gensim format

                query = gensim.matutils.corpus2csc([query], self.index.shape[1], dtype=self.index.dtype)

        # compute cosine similarity against every other document in the collection

        #result = self.index * query.tocsc()  # N x T * T x C = N x C

        result = self.calc_sim(query)

        if result.shape[1] == 1 and not is_corpus:

            # for queries of one document, return a 1d array

            result = result.toarray().flatten()

        elif self.maintain_sparsity:

            # avoid converting to dense array if maintaining sparsity

            result = result.T

        else:

            # otherwise, return a 2d matrix (#queries x #index)

            result = result.toarray().T

        return result
corpus_csr = gensim.matutils.corpus2csc((word_dic.doc2bow(ee) for ee in itertools.islice(sentences, None)), num_terms=len(word_dic)).T

corpus_csr.shape
mysim = MySparseMatrixSimilarity(corpus_csr, num_features=len(word_dic), tfidf=tfidf)

mysim
mysim.num_best = 20

mysim.method = 'WT_SMART2'

mysim.SLOPE = 0

query = word_dic.doc2bow(['finance'])

query
'''

get documents

'''

res = mysim[query]

res
# most similar document

# (10695, 0.12890419363975525)

' '.join([word_dic[ee-3] for ee in docs[res[0][0]]])
# (4419, 0.1041804701089859)

' '.join([word_dic[ee-3] for ee in docs[res[1][0]]])
'''another example'''

query = word_dic.doc2bow(['mcgrath', 'rentcorp'])



res = mysim[query]

res
# most similar

# (0, 0.26121574640274048)

doc_id = 0

' '.join([word_dic[ee-3] for ee in docs[doc_id]])
doc_id = 0

' '.join([word_dic[ee-3] for ee in docs[doc_id]])
# TF => bad result

res = mysim.getCorpusByDoc(doc_id, method='WT_TF')

res2 = gensim.matutils.scipy2sparse(res)

res2.sort(key=lambda x: x[1], reverse=True)

for idx1, v in res2:

    print(idx1, word_dic[idx1], v)
# TFIDF => so so

res = mysim.getCorpusByDoc(doc_id, method='WT_TFIDF')

res2 = gensim.matutils.scipy2sparse(res)

res2.sort(key=lambda x: x[1], reverse=True)

for idx1, v in res2:

    print(idx1, word_dic[idx1], v)
# SMART2 => good result

res = mysim.getCorpusByDoc(doc_id, method='WT_SMART2')

res2 = gensim.matutils.scipy2sparse(res)

res2.sort(key=lambda x: x[1], reverse=True)

for idx1, v in res2:

    print(idx1, word_dic[idx1], v)
# (10695, 0.12890419363975525)

doc_id = 10695



res = mysim.getCorpusByDoc(doc_id, method='WT_SMART2')

res2 = gensim.matutils.scipy2sparse(res)

res2.sort(key=lambda x: x[1], reverse=True)

for idx1, v in itertools.islice(res2, 20):

    print(idx1, word_dic[idx1], v)
# (4419, 0.1041804701089859)

doc_id = 4419



res = mysim.getCorpusByDoc(doc_id, method='WT_SMART2')

res2 = gensim.matutils.scipy2sparse(res)

res2.sort(key=lambda x: x[1], reverse=True)

for idx1, v in itertools.islice(res2, 20):

    print(idx1, word_dic[idx1], v)
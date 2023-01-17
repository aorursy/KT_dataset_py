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
combined_news_djia = pd.read_csv('../input/Combined_News_DJIA.csv')

djia_table = pd.read_csv('../input/DJIA_table.csv')

redditnews = pd.read_csv('../input/RedditNews.csv')
combined_news_djia.head()
djia_table.head()
redditnews.head()
import matplotlib

import matplotlib.pylab as plt

matplotlib.style.use("ggplot")



djia_table[['Date', 'Close']].plot(color='blue', figsize=(10,7))

plt.show()
data = combined_news_djia.merge(djia_table[['Date', 'Close']], how='left', on='Date')

data['Close_diff'] = data['Close'].diff()

data.head()
data.dtypes
import re



def analyzer(sentence):

    sentence = str(sentence)

    sentence = sentence.lower()

    sentence = re.sub(re.compile('[\s\t\r\n]'), ' ', sentence)

    sentence = re.sub(re.compile('[^a-z\s]'), ' ', sentence)

    sentence = re.sub(re.compile('[\s]+'), ' ', sentence)

    return sentence



dates, closes, close_diffs,close_abs_diffs,  labels, news = [], [], [], [], [], []

for i, row in data.iterrows():

    dates.append(row['Date'])

    closes.append(row['Close'])

    close_diffs.append(row['Close_diff'])

    close_abs_diffs.append(abs(row['Close_diff']))

    labels.append(row['Label'])

    news.append(' '.join(row['Top1':'Top25'].apply(analyzer).as_matrix()))
len(dates)
stop_words = ['b', 'a', 'am' ,'an' ,'and', 'are', 'as', 'at', 'be', 'but', 'by', 'did', 'do', 'does', 'for', 'from', 'had', 'has', 'have', 'how', 'i', 'if', 'in', 'is', 'it', 'its', 'me', 'my', 'no', 'not', 'of', 'on', 'or', 'to', 'over', 'same', 'so', 'some', 'such', 'than', 'that', 'the', 'then', 'there', 'they', 'this', 'to', 'too', 'very', 'was', 'we', 'what', 'when', 'where', 'which', 'who', 'why', 'with', 'would', 'you', 'your']



def analyzer2(sentence):

    tmp = []

    words = sentence.split(' ')

    for word in words:

        if word not in stop_words and len(word) > 1:

            tmp.append(word)

    return ' '.join(tmp).strip()

    

news_tmp = []

for sentence in news:

    news_tmp.append(analyzer2(sentence))

news = news_tmp

len(news)
# BoW, tf-idf, Word2Vec, SCDV



features_num = 200

min_word_count = 10

context = 5

downsampling = 1e-3

epoch_num = 10
# BoW



import time

from sklearn.feature_extraction.text import CountVectorizer



st = time.time()

corpus = news

count_vectorizer = CountVectorizer(min_df=min_word_count, binary=True)

bows = count_vectorizer.fit_transform(corpus)

ed = time.time()

print(ed-st)



bows.shape
from sklearn.manifold import TSNE



st = time.time()

tsne_bow = TSNE(n_components=2).fit_transform(bows.toarray())

ed = time.time()

print(ed-st)
tsne_bow_df = pd.DataFrame({

    'x': tsne_bow[:, 0],

    'y': tsne_bow[:, 1],

    'label': labels,

    'close_diff': close_diffs,

    'close_abs_diff': close_abs_diffs

})
tsne_bow_df.plot.scatter(x='x', y='y', c='label', cmap='bwr', figsize=(15, 10), s=20)

plt.show()
tsne_bow_df.plot.scatter(x='x', y='y', c='close_diff', cmap='bwr', figsize=(15, 10), s=20)

plt.show()
tsne_bow_df.plot.scatter(x='x', y='y', c='close_abs_diff', cmap='bwr', figsize=(15, 10), s=20)

plt.show()
# tf-idf 



from sklearn.feature_extraction.text import TfidfVectorizer



st = time.time()

corpus = news

tfidf_vectorizer = TfidfVectorizer(min_df=min_word_count)

tfidfs = tfidf_vectorizer.fit_transform(corpus)

ed = time.time()

print(ed-st)



tfidfs.shape
st = time.time()

tsne_tfidf = TSNE(n_components=2).fit_transform(tfidfs.toarray())

ed = time.time()

print(ed-st)
tsne_tfidf_df = pd.DataFrame({

    'x': tsne_tfidf[:, 0],

    'y': tsne_tfidf[:, 1],

    'label': labels,

    'close_diff': close_diffs,

    'close_abs_diff': close_abs_diffs

})
tsne_tfidf_df.plot.scatter(x='x', y='y', c='label', cmap='bwr', figsize=(15, 10), s=20)

plt.show()
tsne_tfidf_df.plot.scatter(x='x', y='y', c='close_diff', cmap='bwr', figsize=(15, 10), s=20)

plt.show()
tsne_tfidf_df.plot.scatter(x='x', y='y', c='close_abs_diff', cmap='bwr', figsize=(15, 10), s=20)

plt.show()
# Word2Vec



from gensim.models import Word2Vec



st = time.time()

corpus = []

for sentence in news:

    corpus.append(sentence.split(' '))

word2vecs = Word2Vec(

    sentences = corpus,

     iter = epoch_num,

     size = features_num,

     min_count = min_word_count,

     window = context,

     sample = downsampling,

)

#path = './{}_features_word2vecs'.format(features_num)

#word2vecs.save(path)

#word2vecs = Word2Vec.load(path)

ed = time.time()

print(ed-st)
def to_word2vec_avg(sentence):

    score = np.zeros(features_num, dtype=np.float32)

    sentences = sentence.split(' ')

    for word in sentences:

        if word in word2vecs:

            score += word2vecs[word]

    score /= len(sentences)

    return score



word2vec_avgs = []

for sentence in news:

    word2vec_avgs.append(to_word2vec_avg(sentence))

word2vec_avgs = np.array(word2vec_avgs)

word2vec_avgs.shape
st = time.time()

tsne_word2vec_avgs = TSNE(n_components=2).fit_transform(word2vec_avgs)

ed = time.time()

print(ed-st)
tsne_word2vec_avg_df = pd.DataFrame({

    'x': tsne_word2vec_avgs[:, 0],

    'y': tsne_word2vec_avgs[:, 1],

    'label': labels,

    'close_diff': close_diffs,

    'close_abs_diff': close_abs_diffs

})
tsne_word2vec_avg_df.plot.scatter(x='x', y='y', c='label', cmap='bwr', figsize=(15, 10), s=20)

plt.show()
tsne_word2vec_avg_df.plot.scatter(x='x', y='y', c='close_diff', cmap='bwr', figsize=(15, 10), s=20)

plt.show()
tsne_word2vec_avg_df.plot.scatter(x='x', y='y', c='close_abs_diff', cmap='bwr', figsize=(15, 10), s=20)

plt.show()
# SCDV-Word2Vec

# https://arxiv.org/pdf/1612.06778.pdf



from sklearn.mixture import GaussianMixture



st = time.time()

word_vectors = word2vecs.wv.syn0

clusters_num = 60

gmm = GaussianMixture(n_components=clusters_num, covariance_type='tied', max_iter=50)

gmm.fit(word_vectors)

ed = time.time()

print(ed-st)
idf_dic = dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectorizer._tfidf.idf_))

assign_dic = dict(zip(word2vecs.wv.index2word, gmm.predict(word_vectors)))

soft_assign_dic = dict(zip(word2vecs.wv.index2word, gmm.predict_proba(word_vectors)))
st = time.time()

word_topic_vecs = {}

for word in assign_dic:

    word_topic_vecs[word] = np.zeros(features_num*clusters_num, dtype=np.float32)

    for i in range(0, clusters_num):

        try:

            word_topic_vecs[word][i*features_num:(i+1)*features_num] = word2vecs[word]*soft_assign_dic[word][i]*idf_dic[word]

        except:

            continue

ed = time.time()

print(ed-st)
scdvs = np.zeros((len(news), clusters_num*features_num), dtype=np.float32)



a_min = 0

a_max = 0



for i, sentence in enumerate(news):

    tmp = np.zeros(clusters_num*features_num, dtype=np.float32)

    words = sentence.split(' ')

    for word in words:

        if word in word_topic_vecs:

            tmp += word_topic_vecs[word]

    norm = np.sqrt(np.sum(tmp**2))

    if norm > 0:

        tmp /= norm

    a_min += min(tmp)

    a_max += max(tmp)

    scdvs[i] = tmp



p = 0.04

a_min = a_min*1.0 / len(news)

a_max = a_max*1.0 / len(news)

thres = (abs(a_min)+abs(a_max)) / 2

thres *= p



scdvs[abs(scdvs) < thres] = 0

scdvs.shape
st = time.time()

tsne_scdv = TSNE(n_components=2).fit_transform(scdvs)

ed = time.time()

print(ed-st)
tsne_scdv_df = pd.DataFrame({

    'x': tsne_scdv[:, 0],

    'y': tsne_scdv[:, 1],

    'label': labels,

    'close_diff': close_diffs,

    'close_abs_diff': close_abs_diffs

})
tsne_scdv_df.plot.scatter(x='x', y='y', c='label', cmap='bwr', figsize=(15, 10), s=20)

plt.show()
tsne_scdv_df.plot.scatter(x='x', y='y', c='close_diff', cmap='bwr', figsize=(15, 10), s=20)

plt.show()
tsne_scdv_df.plot.scatter(x='x', y='y', c='close_abs_diff', cmap='bwr', figsize=(15, 10), s=20)

plt.show()
!echo 'uh...'
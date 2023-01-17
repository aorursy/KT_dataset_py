# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
from collections import Counter
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
import re
import sys
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import word2vec

from sklearn.manifold import TSNE
from sklearn import metrics
import pandas as pd 
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_similarity_score
cv = CountVectorizer()
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
stop = set(stopwords.words("english"))
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/obama-white-house.csv"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/obama-white-house.csv",nrows=1000)

data.head(100)
stopwords = set(STOPWORDS)
wordcloud_title = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data['title']))
plt.imshow(wordcloud_title)
plt.axis('off')
plt.title("Title")
print(wordcloud_title.words_)
stopwords = set(STOPWORDS)
wordcloud_content = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data['content']))
plt.imshow(wordcloud_content)
plt.axis('off')
plt.title('Content')
words = wordcloud_content.words_
words_top10 = list(words.keys())[1:11]

word_dates = []
content = data['content']
for word in words_top10:
    dates = []
    
    for i in range(content.size):
        string = content[i]
        if string.find(word) != -1:
            dates.append(data['document_date'][i])
    
    ##[datetime.date(x, '%Y-%m-%d') for x in dates] - ось тут виникають проблеми, дати не конвертуються
    
    word_dates.append(dates)
%%timeit
def cleaning(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    return s
data['content'] = [cleaning(s) for s in data['content']]
data['title'] = [cleaning(s) for s in data['title']]


#StopWordsRemove

#data['content'] = data.apply(lambda row: nltk.word_tokenize(row['content']),axis=1)
#data['title'] = data.apply(lambda row: nltk.word_tokenize(row['title']),axis=1)

#data['content'] = data['content'].apply(lambda x : [item for item in x if item not in stop])
#data['title'] = data['title'].apply(lambda x : [item for item in x if item not in stop])
vectorizer = TfidfVectorizer(stop_words='english',use_idf=True)
model = vectorizer.fit_transform(data['content'].str.upper())
km = KMeans(n_clusters=5,init='k-means++',max_iter=200,n_init=1)

k=km.fit(model)
terms = vectorizer.get_feature_names()
order_centroids = km.cluster_centers_.argsort()[:,::-1]
for i in range(5):
    print("cluster of words %d:" %i)
    for ind in order_centroids[i,:10]:
        print(' %s' % terms[ind])
    print() 
    
def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['title', 'content']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus

corpus = build_corpus(data)        
corpus[0:2]

model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=400, workers=4)
model.wv['states']
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


tsne_plot(model)
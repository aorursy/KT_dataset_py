# GoogleNews-vectors-negative300
!wget --header="Host: doc-0k-64-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" --header="Accept-Language: en-US,en;q=0.9" --header="Cookie: AUTH_sl8scr3r78dtlbjs5mvoco4ilf6h1a3b=09929041593969215305|1542391200000|409ktbjcsj48hp7irn60farm0jmshpq0; NID=144=iUZLDGG1-D1dE5x5RLpA2pGNWimx6WHQ-g4sz0bLPUDSHeHy6sI6XhiSEJ2z4YqrP-S_-rirUP-2C75qLXmC01uAkbOsJZvJPogLAa1uKW1i2UjM8beCIYenF3TWqmPWZLW38-CRyh-9yKI_elj6rjlnuezxRYPHc6BmXZe5RDA" --header="Connection: keep-alive" "https://doc-0k-64-docs.googleusercontent.com/docs/securesc/sr3ne958tilodvhhec79bitjd2u2bnlr/j8ko7ih5d7a02ae9qfsd9j8545eqhnlp/1542391200000/06848720943842814915/09929041593969215305/0B7XkCwpI5KDYNlNUTTlSS21pQmM?e=download" -O "GoogleNews-vectors-negative300.bin.gz" -c
# loading libraries and data

%matplotlib inline

import sqlite3                          # for sql database
import pandas as pd
import numpy as np
import nltk                             # nltk:- Natural Language Processing Toolkit
import string
import re
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import pickle
def saveindisk(obj,filename):
    pickle.dump(obj,open(filename+".p","wb"), protocol=4)
def openfromdisk(filename):
    temp = pickle.load(open(filename+".p","rb"))
    return temp

con = sqlite3.connect("../input/database.sqlite")

# Filtering only positive and negative reviews that is
# not taking into consideration those reviews with score = 3
df = pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3
""", con)


# Give reviews with score > 3 to be positive rating and reviews with a score < 3 as a negative
def polarity(x):
    if x < 3:
        return 'Negative'
    else:
        return 'Positive'
df["Score"] = df["Score"].map(polarity) # map is use to assign in all the Score
df.head() # top 5 values
df.describe()
df.shape
df['Score'].size
df['Score'].value_counts()
df.duplicated(subset={"UserId","ProfileName","Time","Text"}).value_counts() # checking duplicates
display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND UserId="AR5J8UI46CURR"
ORDER BY ProductID
""", con)
display


df1 =  df.drop_duplicates(subset={"UserId","ProfileName","Time","Text"},keep="first") # Deleting all the duplicates
size_diff = df1['Id'].size/df['Id'].size
filtered_data2 = df1[df1.HelpfulnessNumerator <= df1.HelpfulnessDenominator]
# A regular expression (or RE) specifies a set of strings that matches it; the functions in this module let you check if a particular string matches a given regular expression (or if a given regular expression matches a particular string, which comes down to the same thing).
import re
# cleaning html symbols from the sentence
def cleanhtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext

# cleaning punctuations from the sentence
def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stop = stopwords.words('english')
print(stop)
from nltk.stem import SnowballStemmer # Stemmers remove morphological affixes from words, leaving only the word stem.
snow = SnowballStemmer('english') 
print(snow.stem('tasty'))
i = 0
string1 = ' '
final_string = []
all_positive_words = []                   # store words from +ve reviews here
all_negative_words = []                   # store words from -ve reviews here.
s = ''

for sent in filtered_data2['Text'].values:
    filtered_sentence = []
    sent = cleanhtml(sent)
    sent = cleanpunc(sent)
    for w in sent.split():
        if((w.isalpha()) and (len(w)>2)):  
            if(w.lower() not in stop):    # If it is a stopword
                s = (snow.stem(w.lower())).encode('utf8')
                filtered_sentence.append(s)
                if (filtered_data2['Score'].values)[i] == 'Positive':
                    all_positive_words.append(s)
                if(filtered_data2['Score'].values)[i] == 'Negative':
                    all_negative_words.append(s)
            else:
                continue
        else:
            continue 
    string1 = b" ".join(filtered_sentence) 
    final_string.append(string1)
    i += 1
i = 0
string1 = ' '
final_string_nostem = []
s = ''

for sent in filtered_data2['Text'].values:
    filtered_sentence=[]
    sent = cleanhtml(sent)
    sent = cleanpunc(sent)
    for w in sent.split():
        if((w.isalpha()) and (len(w)>2)):  
            if(w.lower() not in stop):
                s = w.lower().encode('utf8')
                filtered_sentence.append(s)
            else:
                continue
        else:
            continue 
    string1 = b" ".join(filtered_sentence)     
    final_string_nostem.append(string1)
    i += 1
from collections import Counter
print("Number of positive words:",len(all_positive_words))
print("Number of negative words:", len(all_negative_words))
filtered_data2['CleanedText'] = final_string
filtered_data2['CleanedText_NoStem'] = final_string_nostem
filtered_data2.head(3)

filtered_data2['CleanedText_NoStem'][1]
from sklearn.feature_extraction.text import CountVectorizer
filtered_data2['CleanedText'].values
uni_gram = CountVectorizer()
uni_gram_vectors = uni_gram.fit_transform(filtered_data2['CleanedText'].values)
uni_gram_vectors.shape[1]
uni_gram_vectors[0]
type(uni_gram_vectors)
from sklearn.decomposition import TruncatedSVD

tsvd_uni = TruncatedSVD(n_components=2)
tsvd_uni_vec = tsvd_uni.fit_transform(uni_gram_vectors)
tsvd_uni.explained_variance_ratio_[:].sum()
# Perplexity = 40

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, tsvd_uni_vec.shape[0]), n_samples)
sample_features = tsvd_uni_vec[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=40)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
# Perplexity = 30

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, tsvd_uni_vec.shape[0]), n_samples)
sample_features = tsvd_uni_vec[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=20)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
bi_gram = CountVectorizer(ngram_range=(1,2))
bi_gram_vectors = bi_gram.fit_transform(filtered_data2['CleanedText'].values)
bi_gram_vectors.shape 
type(bi_gram_vectors)
from sklearn.decomposition import TruncatedSVD
sample_points = filtered_data2.sample(1100)

bi_gram = CountVectorizer(ngram_range=(1,2))
bi_gram_vectors = bi_gram.fit_transform(sample_points['CleanedText'])
tsvd_bi = TruncatedSVD(n_components=2)
tsvd_bi_vec = tsvd_bi.fit_transform(bi_gram_vectors)
tsvd_bi.explained_variance_ratio_[:].sum()
# Perplexity = 30

from sklearn.manifold import TSNE
from time import time
import random

n_samples = 1000
sample_cols = random.sample(range(1, tsvd_bi_vec.shape[0]), n_samples)
sample_features = tsvd_bi_vec[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=30)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
# Perplexity = 20

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, tsvd_bi_vec.shape[0]), n_samples)
sample_features = tsvd_bi_vec[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=20)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,2))
tfidf_vec = tfidf.fit_transform(filtered_data2['CleanedText'])
tfidf_vec.shape
features = tfidf.get_feature_names()
features[190000:190010]
def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ind = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ind]
    df = pd.DataFrame(top_feats,columns = ['feature', 'tfidf'])
    return df
top_tfidfs = top_tfidf_feats(tfidf_vec[1000,:].toarray()[0],features, 20)
from sklearn.decomposition import TruncatedSVD

tsvd_tfidf = TruncatedSVD(n_components=2)
tsvd_tfidf_vec = tsvd_tfidf.fit_transform(tfidf_vec)
tsvd_tfidf.explained_variance_ratio_[:].sum() 
# Perplexity = 20

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, tsvd_tfidf_vec.shape[0]), n_samples)
sample_features = tsvd_tfidf_vec[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=20)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
# Perplexity = 30 

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, tsvd_tfidf_vec.shape[0]), n_samples)
sample_features = tsvd_tfidf_vec[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=30)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
# Perplexity = 40

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, tsvd_tfidf_vec.shape[0]), n_samples)
sample_features = tsvd_tfidf_vec[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=40)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
final_string = []
for sent in filtered_data2['CleanedText'].values:
    sent = str(sent)
    sentence = []
    for word in sent.split():
        sentence.append(word)
    final_string.append(sentence)
import gensim # In gensim a corpus is simply an object which, when iterated over, returns its documents represented as sparse vectors.
w2v_model = gensim.models.Word2Vec(final_string,min_count=5,size=50, workers=-1)
w2v_vocub = w2v_model.wv.vocab
len(w2v_vocub)
w2v_model.wv.most_similar('tast')
avg_vec = []
for sent in final_string:
    cnt = 0
    sent_vec = np.zeros(50)
    for word in sent:
        try:
            wvec = w2v_model.wv[word]
            sent_vec += wvec
            cnt += 1
        except: 
            pass
    sent_vec /= cnt
    avg_vec.append(sent_vec)
avg_vec = np.array(avg_vec)
avg_vec.shape
avg_vec
np.any(np.isnan(avg_vec))
np.all(np.isfinite(avg_vec))
col_mean = np.nanmean(avg_vec, axis=0)
inds = np.where(np.isnan(avg_vec))
inds
avg_vec[inds] = np.take(col_mean, inds[1])

np.any(np.isnan(avg_vec))
from sklearn import preprocessing
avg_vec_norm = preprocessing.normalize(avg_vec)
# Perplexity = 30

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, avg_vec.shape[0]), n_samples)
sample_features = avg_vec[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=30)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
# Perplexity = 20

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, avg_vec_norm.shape[0]), n_samples)
sample_features = avg_vec_norm[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=20)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
from gensim.models import KeyedVectors
w2v_model_google = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
w2v_model_google.wv["word"].size
import random
avg_vec_google = []
datapoint = 3000
sample_cols = random.sample(range(1, datapoint), 1001)

for sent in filtered_data2['CleanedText_NoStem'].values[sample_cols]:
    cnt = 0
    sent_vec = np.zeros(300)
    sent = sent.decode("utf-8") 
    for word in sent.split():
        try:
            wvec = w2v_model_google.wv[word]
            sent_vec += wvec
            cnt += 1
        except: 
            pass
    sent_vec /= cnt
    avg_vec_google.append(sent_vec)
avg_vec_google = np.array(avg_vec_google)
np.any(np.isnan(avg_vec_google))
np.all(np.isfinite(avg_vec_google))
col_mean = np.nanmean(avg_vec_google, axis=0)
inds = np.where(np.isnan(avg_vec_google)) 
inds
avg_vec_google[inds] = np.take(col_mean, inds[1])

avg_vec_google
from sklearn import preprocessing
avg_vec_google_norm = preprocessing.normalize(avg_vec_google)
# Perplexity = 20

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, avg_vec_google.shape[0]), n_samples)
sample_features = avg_vec_google[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=20)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
# Perplexity = 30

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, avg_vec_google.shape[0]), n_samples)
sample_features = avg_vec_google[sample_cols]
sample_class = filtered_data2['Score'][sample_cols]

sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=30)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
# Perplexity = 50

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, avg_vec_google.shape[0]), n_samples)
sample_features = avg_vec_google[sample_cols]
sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=50)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
datapoints = 3000
sample_cols = random.sample(range(1, datapoints), 1001)
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,2))
tfidf_vec_ns = tfidf.fit_transform(filtered_data2['CleanedText_NoStem'].values[sample_cols])
tsvd_tfidf_ns = TruncatedSVD(n_components=2)
tsvd_tfidf_vec_ns = tsvd_tfidf_ns.fit_transform(tfidf_vec_ns)
features = tfidf.get_feature_names()
tfidf_w2v_vec_google = []
review = 0

for sent in filtered_data2['CleanedText_NoStem'].values[sample_cols]:
    cnt = 0 
    weighted_sum  = 0
    sent_vec = np.zeros(300)
    sent = sent.decode("utf-8") 
    for word in sent.split():
        try:
            wvec = w2v_model_google.wv[word] 
            tfidf = tfidf_vec_ns[review,features.index(word)]
            sent_vec += (wvec * tfidf)
            weighted_sum += tfidf
        except:
            pass
    sent_vec /= weighted_sum
    tfidf_w2v_vec_google.append(sent_vec)
    review += 1
len(tfidf_w2v_vec_google)
len(tfidf_w2v_vec_google[0])
tfidf_w2v_vec_google[0]
from sklearn import preprocessing
tfidf_w2v_vec_google_norm = preprocessing.normalize(tfidf_w2v_vec_google)
# Perplexity = 20

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, tfidf_w2v_vec_google_norm.shape[0]), n_samples)
sample_features = tfidf_w2v_vec_google_norm[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=20)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
# Perplexity = 35

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, tfidf_w2v_vec_google_norm.shape[0]), n_samples)
sample_features = tfidf_w2v_vec_google_norm[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=35)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()
# Perplexity = 50

from sklearn.manifold import TSNE
import random

n_samples = 1000
sample_cols = random.sample(range(1, tfidf_w2v_vec_google_norm.shape[0]), n_samples)
sample_features = tfidf_w2v_vec_google_norm[sample_cols]

sample_class = filtered_data2['Score'][sample_cols]
sample_class = sample_class[:,np.newaxis]
model = TSNE(n_components=2,random_state=0,perplexity=50)

embedded_data = model.fit_transform(sample_features)
final_data = np.concatenate((embedded_data,sample_class),axis=1)
tsne_data = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","label"])

sns.FacetGrid(tsne_data,hue="label",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
plt.show()

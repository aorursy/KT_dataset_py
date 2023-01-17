#We shall import some stuff here



import pandas as pd

import numpy as np

import re

import string



from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag

from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score



import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
# Initializing a lemmatizer

lemmatizer = WordNetLemmatizer()
# Lets open the file and and extract all the content line by line into a list

with open('../input/bbchealth.txt') as file:

    contents = file.readlines()

    

# Fancy regex for extracting data. Because I'm awesome and I refuse to split it.

regex = re.compile(r'(?P<tweet_id>.*)\|(?P<date>.*)\|(?P<news>.*)\s?(?P<link>http://.*)')
# I hope the regex words

re.search(regex, contents[0]).groupdict()



# And it worked. Cool
# You all know this. Don't pretend that you don't.

df = pd.DataFrame(columns=['tweet_id', 'date', 'news', 'link'])



# Self explanatory

for line in contents:

    line_data = re.search(regex, line).groupdict()

    df = df.append(line_data, ignore_index=True)
# Our dataset looks like this. Not too shabby.

df.head()
# Lets define some function to normalize, lemmatize the line and getting POS tags. 

# Normalizing, the following methods in order -- lower, strip white spaces, remove punctiations and 

# digits, and finally word tokenizing



def normalize(line):

    line = line.lower().strip()

    line = ''.join([char for char in line if char not in string.punctuation+string.digits])

    return word_tokenize(line)



# Lemmatizing each word in each line

def lemmatize_sent(line_tokens):

    return list(map(lemmatizer.lemmatize, line_tokens))



# POS tagging, This one returns just the POS tags in order.

def tokens(line_tokens):

    word_tags = pos_tag(line_tokens)

    return list(zip(*word_tags))[1]
# Let's apply all those functions to the columns.

df['news_tokens'] = df['news'].map(normalize)

df['corresponding_tags'] = df['news_tokens'].map(tokens)

df['lemmatized_news_tokens'] = df['news_tokens'].map(lemmatize_sent)
# Dropping unnecessary columns, that I think I don't need.

df.drop(labels=['link'], inplace=True, axis=1)

df.drop(labels=['tweet_id'], inplace=True, axis=1)
# Let's see what our dataset looks like again

df.head()
# Now that we've cleaned and prepared our dataset, let's do some transformation

# We'll create a tfidf matrix of all the documents. 

# Tfidf is generally used in document similarity. Because the dot product followed by division by the their

# normals give you their cosine similarity.

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(df['lemmatized_news_tokens'].map(lambda x: ' '.join(x)).tolist())
# Let's do some actual machine learning now. We'll be doing KMeans for now.

# Let's take the default value for number of clusters(n_clusters=8).

# because we don't know any crap.

# And later use elbow method and silhoutte score to find the right number of 

# clusters

kmeans = KMeans()

kmeans.fit(tfidf_matrix)
# Yup, right number of labels.

np.unique(kmeans.labels_)
tfidf_matrix
# Now, plotting time. But we can't just plot a sparse matrix of size 3929x3955.

# That's stupid. So, let's trim down the dimensions to 2.

# This is going to lose some information, but do we have a choice?

pca = PCA(n_components=2)

pca_matrix = pca.fit_transform(tfidf_matrix.A)
# Let's plot the cluster centers

plt.figure(figsize=(5, 5))

cluster_centers = pca.fit_transform(kmeans.cluster_centers_)

sns.scatterplot(x=cluster_centers[:,0], y=cluster_centers[:,1], hue=np.unique(kmeans.labels_))

    

plt.xlim((-0.4, 0.4))

plt.ylim((-0.4, 0.4))



# Oh well. Horrible.
# Plotting the whole data, we get the following graph.

plt.figure(figsize=(10, 10))

sns.scatterplot(x=pca_matrix[:,0], y=pca_matrix[:,1], hue=kmeans.labels_)
# Did you just think we're gonna stop at 2D plots?

# If you did, you're wrong buddy. Behold 3D plots.

pca_3d = PCA(n_components=3)

pca_matrix_3d = pca_3d.fit_transform(tfidf_matrix.A)
fig = plt.figure(figsize=(5, 5))

ax = fig.add_subplot(111, projection='3d')

for p, l in zip(pca_matrix_3d, kmeans.labels_):

    plt.scatter(*p, 'bgrcmykw'[l])
# Now, for elbow method. We'll try clustering from k=1 to k=15 and see how well the curve is displayed

x, y = [], []

for k in range(1, 16):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(tfidf_matrix)

    x.append(k)

    y.append(kmeans.inertia_)

    

plt.plot(x, y)
# That suck. Where's the elbow?

# Let's see how sihoutte score goes.

# For those who doesn't know what silhoutte score is,

# The silhoutte score is a value of measure of closeness

# to it's own cluster(cohesion) compared to other clusters(seperation)

# it ranges from [-1, 1]. A high value indicates how well matched is

# an object is to it's cluster.

# For more info, wikipedia is your guy.



# We'll calculate silhoutte scores for the same range as before excpet 1

# Because it'll throw an error if k = 1. Because you know,

# how well a point is matched to it's cluster and there exists one cluster.

# Make sense to you. It doesn't to me.
for k in range(2, 16):

    kmeans = KMeans(n_clusters=k).fit(tfidf_matrix)

    sil_score = silhouette_score(tfidf_matrix, kmeans.labels_)

    print('Number of clusters: {}, Silhoutte Score: {}'.format(k, sil_score))
# That's not nice, the silhoutte score is still going up.

# No matters, now this is where you come in visitor. If you find

# something, please post it in the comment section. It can be 

# suggestions, imrpovements, or errors. 
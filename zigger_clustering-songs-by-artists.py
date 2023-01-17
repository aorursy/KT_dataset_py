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
df = pd.read_csv('../input/songdata.csv')

# remove new lines from lyrics

df['text'] = df.text.apply(lambda x: x.replace('\n',''))
artists = ['Death', 'Xzibit', 'Britney Spears']

num_artists = len(artists)

df_artists = df.loc[df.artist.isin(artists)]

for artist in artists:

    print ('Number of {} songs: {}'.format(artist, len(df_artists[df_artists.artist == artist])))
df_artists.head()
stopwords = stopwords.words('english')

punctuation = string.punctuation

trantab = str.maketrans(punctuation, ' '*len(punctuation))



def process_text(text):

    """Remove punctuation, lower text, tokenize text, remove stop words and stem tokens (words).

    Args:

      text: a string.

    Returns:

      Tokenized text i.e. list of stemmed words. 

    """

    

    # replace all punctuation by blanc spaces

    text = text.translate(trantab)

    

    # lower text

    text = text.lower()

    

    # tokenize text

    word_tokens = word_tokenize(text) 

  

    # remove stop words

    filtered_text = [w for w in word_tokens if not w in stopwords] 

        

    # stemm text

    stemmer = PorterStemmer()

    tokens = [stemmer.stem(t) for t in filtered_text]



    return tokens
def define_vectorizer(additional_stopwords = [], max_df=0.5, min_df=0.1, ngram_range=(1, 1)):

    """Transform texts to Tf-Idf coordinates.

    Args:

      additional_stop_words: addititional stop_words, list of strings.

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

                             stop_words=additional_stopwords,

                             max_df=max_df,

                             min_df=min_df,

                             ngram_range=ngram_range)

    return vectorizer



def compute_tfidf_matrix(corpus, vectorizer):

    """Transform texts to Tf-Idf coordinates.

    Args:

      corpus: list of strings.

      vectorizer: sklearn TfidfVectorizer.

    Returns:

      A sparse matrix in Compressed Sparse Row format with tf-idf scores.

    Raises:

      ValueError: If `corpus` generates empty vocabulary or after pruning no terms remain.

    """

    tfidf_matrix = vectorizer.fit_transform(corpus) # raises ValueError if bad corpus

    return tfidf_matrix



def corpus_features(df, field):

    """

    Returns vectorizer, tfidf_matrix, dist computed on df[field].

    """

    corpus = df[field].values

    vectorizer = define_vectorizer()

    tfidf_matrix = compute_tfidf_matrix(corpus = corpus, vectorizer = vectorizer)

    dist = 1 - cosine_similarity(tfidf_matrix)

    return vectorizer, tfidf_matrix, dist
field = 'text'
vectorizer, tfidf_matrix, dist = corpus_features(df_artists, field)
num_clusters = num_artists
def annotate_labels(labels, vectorizer, tfidf_matrix, num_clusters, num_words = 3):

    """ Label clusters with num_words most common words """

    feature_array = vectorizer.get_feature_names()

    # we create a numpy array here in order to use list indexing list_of_clusters[cluster_texts_index]

    # we choose dtype=object because str type will actually be char (string of length 1)

    list_of_clusters = np.empty(len(labels), dtype=object)

    for i in range(num_clusters):

        cluster_texts_index = np.where(labels==i)[0]

        r = tfidf_matrix[cluster_texts_index]

        tfidf_sorting = np.argsort(r.toarray()).flatten()[::-1]

        top_n = [feature_array[k] for k in tfidf_sorting][:num_words]

        word_label = ' '.join(top_n)

        list_of_clusters[cluster_texts_index] = str(i)+' '+word_label # total_label #str(i)+'

    return list_of_clusters

    

def dict_labels(labels): 

    """ Returns a dictionary with names of clusters as keys and lists of indices of corresponding elements as values """

    clustering = collections.defaultdict(list)

    for i, label in enumerate(labels):

        clustering[label].append(i) 

    return dict(clustering)



def conf_matrix_clusters(df, clusters):

    """ Prints the confusion matrix """

    results = []

    cluster_names = clusters.keys()

    assert df['artist'].nunique() == len(cluster_names), "conf_matrix_clusters() only works for num_clusters == num_artists"

    for k in cluster_names:

        row = df.iloc[clusters[k]].groupby('artist')['song'].nunique().to_dict()

        for artist in artists:

            if artist not in row:

                row[artist] = 0

        results.append(row)

    

    results_df = pd.DataFrame(results, cluster_names).transpose()

    

    f, ax = plt.subplots(figsize=(11, 9))

    sns.heatmap(results_df, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
def kmeans_clustering(vectorizer, tfidf_matrix, num_clusters = 0):

    """Perform KMeans clustering.

    Args:

      vectorizer: vectorizer.

      tfidf_matrix: tfidf_matrix.

      num_clusters: non-negative integer. If 0, automatically chooses the optimal number of clusters using

        silouhettes, otherwise performs clustering with `num_clusters` clusters.

    Returns:

      List `labels` where `labels[i]` is the cluster of `i`-th string in the `corpus`.

    """

    

    feature_array = vectorizer.get_feature_names()

    labels = []

    

    # custom number of clusters

    if num_clusters != 0:  

        # fix random_state for reproducibility

        km_model = KMeans(n_clusters=num_clusters, random_state = 42).fit(tfidf_matrix)

        labels = km_model.labels_

        #clusters = cluster_texts(km_model) 

    # auto - using the maximal silhouette score

    if num_clusters == 0:

        range_n_clusters = range(2,21)

        silhouettes = []

        for n_clusters in range_n_clusters:

            km_model = KMeans(n_clusters=n_clusters).fit(tfidf_matrix)

            labels = km_model.labels_

            silhouettes.append(silhouette_score(tfidf_matrix, labels))

        num_clusters = np.argmax(silhouettes) + range_n_clusters[0]

        km_model = KMeans(n_clusters=num_clusters).fit(tfidf_matrix)

        labels = km_model.labels_

        

    return labels
def hierarchical_clustering(vectorizer, tfidf_matrix, num_clusters = 0):

    """Perform hierarchical clustering.

    Args:

      vectorizer: vectorizer.

      tfidf_matrix: tfidf_matrix.

      num_clusters: non-negative integer. If 0, automatically chooses the optimal number of clusters using

        silouhettes, otherwise performs clustering with `num_clusters` clusters.

    Returns:

      List `labels` where `labels[i]` is the cluster of `i`-th string in the `corpus`.

    """

    feature_array = vectorizer.get_feature_names()

    labels = []

    

    # for hierarchical clustering

    dist = 1 - cosine_similarity(tfidf_matrix)

    linkage_matrix = ward(dist)

        

    # custom number of clusters

    if num_clusters != 0:  

        labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust') - 1 # substract 1 to get numeration from 0

    # auto - using the maximal silhouette score

    if num_clusters == 0:

        range_n_clusters = range(2,21)

        silhouettes = []

        for n_clusters in range_n_clusters:

            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1 # substract 1 to get numeration from 0

            silhouettes.append(silhouette_score(tfidf_matrix, labels))

        num_clusters = np.argmax(silhouettes) + range_n_clusters[0]

        labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust') - 1 # substract 1 to get numeration from 0

    

    return labels
def cluster(clustering_function, vectorizer, tfidf_matrix, num_clusters = 3):

    labels = clustering_function( vectorizer, tfidf_matrix, num_clusters)

    annotated_labels = annotate_labels(labels, vectorizer, tfidf_matrix, num_clusters, num_words = 3)

    dict_cluster_indices = dict_labels(annotated_labels)

    conf_matrix_clusters(df_artists, dict_cluster_indices)

    return annotated_labels
kmeans_labels = cluster(kmeans_clustering, vectorizer, tfidf_matrix, num_clusters = 3)
hierarchical_labels = cluster(hierarchical_clustering, vectorizer, tfidf_matrix, num_clusters = 3)
def viz_clusters(dist, labels):

    

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys = pos[:, 0], pos[:, 1]   

    

    #create data frame that has the result of the MDS plus the cluster numbers and titles

    df_viz = pd.DataFrame(dict(x=xs, y=ys, label=labels, title=df_artists['song'].values)) 

    #group by cluster

    groups = df_viz.groupby('label')





    # set up plot

    fig, ax = plt.subplots(figsize=(21, 15)) # set size

    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling



    # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label

    for name, group in groups:

        label = group.iloc[0]['label']

        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=label, mec='none')

        ax.set_aspect('auto')

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        ax.tick_params(axis='y', which='both', left=False, top=False, labelleft=False)



    ax.legend(numpoints=1)  # show legend with only 1 point



    # add label in x,y position with the label as the film title

    for i in range(len(df_viz)):

        ax.text(df_viz.iloc[i]['x'], df_viz.iloc[i]['y'], df_viz.iloc[i]['title'], size=8)  



    plt.show() #show the plot

    

def plot_dendrogram(vectorizer, dist):

    """Plots a dendogram which shows how hierarchical clustering works.

    Args:

      df: pandas data frame with the column `cluster`.

      field: a column in `df` on which we will perform clustering.

    Returns:

      Nothing. Plots the dendogram.

    Raises:

      AssertionError: If a parameter is wrong.

    """

    

    linkage_matrix = ward(dist)



    fig, ax = plt.subplots(figsize=(15, 30)) # set size

    ax = dendrogram(linkage_matrix, orientation="right");



    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)



    plt.tight_layout()
viz_clusters(dist, kmeans_labels)
viz_clusters(dist, hierarchical_labels)
plot_dendrogram(vectorizer, dist)
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, strip_short, strip_punctuation, stem_text
strip_short2 = lambda x: strip_short(x, minsize=2)

# Pipeline for preprocessing in Genism is different. For simplicity we don't use stemming here.

CUSTOM_FILTERS = [lambda x: x, remove_stopwords, strip_punctuation, strip_short2, stem_text]

tokens = df_artists[field].apply(lambda s: preprocess_string(s, CUSTOM_FILTERS))
dictionary = corpora.Dictionary(tokens)

doc_term_matrix = [dictionary.doc2bow(text) for text in tokens]
import warnings

warnings.filterwarnings('ignore')



# Creating the object for LDA model using gensim library

LDA = gensim.models.ldamodel.LdaModel



# number of topics is the same as for K-Means 

num_topics = num_clusters

# Build LDA model

lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, 

                num_topics=num_topics, 

                alpha=[0.0001] * num_topics, 

                eta=[0.0001] * len(dictionary),

                chunksize=2000,

                passes=4,

                random_state=100,

               )
lda_model.print_topics(num_words=8)
# Visualize the topics

pd.options.display.max_colwidth = 2000

viz = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary, mds='tsne')
pyLDAvis.enable_notebook()

viz
def LDA_clustering(num_words=3):

    

    all_topics = lda_model.get_document_topics(doc_term_matrix, per_word_topics=True)

    

    topics = lda_model.print_topics(num_words=num_words)

    clusters_lda = {}

    cluster_names = {}



    # get named clusters 

    for x in topics:

        cluster_words = ''

        for proba_with_word in x[1].split('+'):

            cluster_words += ' ' + proba_with_word.split('"')[1]

        clusters_lda[str(x[0])+cluster_words] = []

        cluster_names[x[0]] = str(x[0])+cluster_words



    counter = 0

    for doc_topics, word_topics, phi_values in all_topics:

        cluster_with_highest_proba = max(doc_topics, key = lambda item: item[1])[0]

        cluster_name = cluster_names[cluster_with_highest_proba]

        clusters_lda[cluster_name].append(counter)

        counter += 1

        

    return clusters_lda
clusters_lda = LDA_clustering()
conf_matrix_clusters(df_artists, clusters = clusters_lda)
def structured_field_clustering(df, field):

    """

    Fill `cluster` column in dataframe `df` based on `field` column. 

    """

    assert field in df.columns, field+' is not in df columns'

    assert sum(df[field].isna()) == 0, 'There are some undefined values in df.'+field

    df_ri = df.reset_index(drop=True)

    df_ri['cluster'] = 'cluster ' + df_ri[field]

    return df_ri
df_artists_SSL = structured_field_clustering(df_artists, 'artist')

for i in range(len(df_artists_SSL)):

    # leave approximately 30% for training

    if np.random.random() > 0.3:

        df_artists_SSL.loc[i, 'cluster'] = None
def auto_fill_in(df, field):

    """

    Automatically propagate labels in `df`.

    """

    assert field in df.columns, field+' is not in df columns'

    df_ri = df.reset_index(drop=True)

    train_ind = df_ri.loc[~df_ri.cluster.isna()].index

    target_ind = df_ri.loc[df_ri.cluster.isna()].index

    assert len(train_ind) > 0, 'There is no labeled data in df'

    assert len(target_ind) > 0, 'There is no unlabeled data in df'

    train = df_ri.iloc[train_ind]

    target = df_ri.iloc[target_ind]

    

    texts = df_ri[field].apply(lambda x: x.lower().replace('\n', ''))

    vectorizer = define_vectorizer()

    X = compute_tfidf_matrix(texts.values, vectorizer)

    

    X_train = X[train_ind]

    X_target = X[target_ind]

    

    le = LabelEncoder()

    y_train = le.fit_transform(train['cluster'].values)

        

    KNN = KNeighborsClassifier(n_neighbors=1) # any n_neighbors is fine

    KNN.fit(X_train, y_train) 

    

    y_target = KNN.predict(X_target)

    y_cluster = le.inverse_transform(y_target)

    

    df_ri.loc[target_ind, 'cluster'] = y_cluster

    

    return df_ri
df_artists_SSL_filled = auto_fill_in(df_artists_SSL, 'text')
conf_matrix_clusters(df_artists_SSL_filled, dict_labels(df_artists_SSL_filled['cluster']))
df_artists_SSL = structured_field_clustering(df_artists, 'artist')

for i in range(len(df_artists_SSL)):

    # leave approximately 50% for training

    if np.random.random() > 0.5:

        df_artists_SSL.loc[i, 'cluster'] = None
df_artists_SSL_filled = auto_fill_in(df_artists_SSL, 'text')
conf_matrix_clusters(df_artists_SSL_filled, dict_labels(df_artists_SSL_filled['cluster']))
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

from pyspark.ml.feature import StringIndexer

from pyspark.ml import Pipeline

from pyspark.ml.clustering import KMeans

from pyspark.ml.evaluation import ClusteringEvaluator



tokenizer = Tokenizer(inputCol="text", outputCol="words")

hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')

idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) # minDocFreq: remove sparse terms



label_stringIdx = StringIndexer(inputCol = "cluster", outputCol = "label").setHandleInvalid("keep") # keep NaN



pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])



df = spark.read.load("../input/df_artists.csv", format="csv", sep=",", inferSchema="true", header="true")

pipelineFit = pipeline.fit(df)

train_df = pipelineFit.transform(df)

train_tfidf_only = train_df.select('features')



for k in range(2,10):

    # Trains a k-means model.

    kmeans = KMeans().setK(k).setSeed(1)

    model = kmeans.fit(train_tfidf_only)

    # Make predictions

    predictions = model.transform(train_tfidf_only)

    # Evaluate clustering by computing Silhouette score

    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)

    print(str(k)+" Silhouette with squared euclidean distance = " + str(silhouette))





# Shows the result.

centers = model.clusterCenters()

print("Cluster Centers: ")

for center in centers:

    print(center)
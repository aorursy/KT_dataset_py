# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_ent = pd.read_csv('../input/output.csv')

df_hack = pd.read_csv('../input/HackerNoon.csv')
df_ent = df_ent.dropna()
df_hack = df_hack.dropna()
print(len(df_ent) ,len(df_hack))
df_combined = pd.concat([df_ent, df_hack.iloc[:308]]).reset_index()
df_combined['Site'].unique()
df_combined['target'] = df_combined['Site'].apply(lambda x : 1 if x == 'https://entrepreneurshandbook.co/' else 0)
df_combined['target'].value_counts()
# Importing Essential Datasets
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
import re
from functools import reduce

def remove_stopwords(words):
    stop_words = set(stopwords.words("english"))
    return [word for word in words if word not in stop_words]

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def lemmatize_text(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def stem_text(words):
    ps = PorterStemmer()
    return [ps.stem(word) for word in words]
df_combined['Text'] = df_combined['Text'].apply(lambda x : x.lower())
df_combined['Text'] = df_combined['Text'].apply(remove_punctuation)
df_combined['Text'] = df_combined['Text'].apply(word_tokenize)
df_combined['Text'] = df_combined['Text'].apply(remove_stopwords)
df_combined['Text'] = df_combined['Text'].apply(lemmatize_text)

print (df_combined.head())
print (df_combined.tail())
tf_idf_vec = TfidfVectorizer(analyzer = 'word',
                            ngram_range = (1, 2),
                            stop_words = 'english')
tf_idf_vec
list(df_combined['Text'].map(lambda tokens : ' '.join(tokens)))[:5]
tf_idf = tf_idf_vec.fit_transform(list(df_combined['Text'].map(lambda x : ' '.join(x))))
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 3, random_state = 1999, n_iter = 50)
svd_tfidf = svd.fit_transform(tf_idf)
print("Dimensionality of LSA space: {}".format(svd_tfidf.shape))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(16,12))

ax = Axes3D(fig)
ax.scatter(svd_tfidf[:,0],
          svd_tfidf[:,1],
          svd_tfidf[:,2],
          c = df_combined.target.values,
          cmap = plt.cm.winter_r,
          s = 4,
          edgecolor = 'none',
          marker = 'o')
# plt.title("Semantic Tf-Idf-SVD reduced plot of Sincere-Insincere data distribution")
plt.xlabel("First dimension")
plt.ylabel("Second dimension")
plt.legend()
plt.xlim(0.0, 0.20)
plt.ylim(-0.2,0.4)
plt.show()
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components = 2,
                  verbose = 1,
                  random_state = 1999,
                  n_iter = 10000,
                  learning_rate = 100)
tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
tsne_tfidf_df = pd.DataFrame(data=tsne_tfidf, columns=["x", "y"])
# tsne_tfidf_df["qid"] = train_rebal["qid"].values
tsne_tfidf_df["text"] = df_combined["Text"].values
tsne_tfidf_df["target"] = df_combined['target'].values
tsne_tfidf_df.head()
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook, reset_output
from bokeh.palettes import d3
import bokeh.models as bmo
from bokeh.io import save, output_file

# init_notebook_mode(connected = True)
# color = sns.color_palette("Set2")
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
output_notebook()
plot_tfidf = bp.figure(plot_width = 800, plot_height = 700, 
                       title = "T-SNE applied to Tfidf_SVD space",
                       tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                       x_axis_type = None, y_axis_type = None, min_border = 1)

# colormap = np.array(["#6d8dca", "#d07d3c"])
colormap = np.array(['red', 'blue'])

# palette = d3["Category10"][len(tsne_tfidf_df["asset_name"].unique())]
source = ColumnDataSource(data = dict(x = tsne_tfidf_df["x"], 
                                      y = tsne_tfidf_df["y"],
                                      color = colormap[tsne_tfidf_df["target"] - 1],
                                      author = df_combined['Author'],
                                      text = tsne_tfidf_df["text"],
                                      target = tsne_tfidf_df["target"]))

plot_tfidf.scatter(x = "x", 
                   y = "y", 
                   color="color",
                   legend = "target",
                   source = source,
                   alpha = 0.7)

hover = plot_tfidf.select(dict(type = HoverTool))
hover.tooltips = {"text" : "@text", 
                  "target" : "@target",
                  'authour' : '@author'
                 }

show(plot_tfidf)
# Hitting Perlexity 5
tsne_model_5 = TSNE(perplexity=5,
                  n_components=2,
                  verbose=1,
                  random_state=1999,
                  n_iter=10000,
                  learning_rate = 100)

tsne_tfidf_5 = tsne_model_5.fit_transform(svd_tfidf)
# Creating a Dataframe for Perplexity=5
tsne_tfidf_df_5 = pd.DataFrame(data=tsne_tfidf_5, columns=["x5", "y5"])
tsne_tfidf_df_5["target"] = df_combined["target"].values

plt.figure(figsize=(14,8))
plt.scatter(tsne_tfidf_df_5.x5, 
            tsne_tfidf_df_5.y5, 
            alpha=1,
            c=tsne_tfidf_df_5.target,
            cmap=plt.cm.coolwarm)
plt.title("T-SNE plot in SVD space (perplexity=5)")
plt.legend()
plt.show()
# Hitting Perlexity 5
tsne_model_25 = TSNE(perplexity=25,
                  n_components=2,
                  verbose=1,
                  random_state=1999,
                  n_iter=10000,
                  learning_rate = 100)

tsne_tfidf_25 = tsne_model_25.fit_transform(svd_tfidf)
# Creating a Dataframe for Perplexity=5
tsne_tfidf_df_25 = pd.DataFrame(data=tsne_tfidf_25, columns=["x25", "y25"])
tsne_tfidf_df_25["target"] = df_combined["target"].values

plt.figure(figsize=(14,8))
plt.scatter(tsne_tfidf_df_25.x25, 
            tsne_tfidf_df_25.y25, 
            alpha=1,
            c=tsne_tfidf_df_25.target,
            cmap=plt.cm.coolwarm)
plt.title("T-SNE plot in SVD space (perplexity=25)")
plt.legend()
plt.show()
# Hitting Perlexity 50
tsne_model_50 = TSNE(perplexity=50,
                  n_components=2,
                  verbose=1,
                  random_state=1999,
                  n_iter=10000,
                  learning_rate = 100)

tsne_tfidf_50 = tsne_model_50.fit_transform(svd_tfidf)
# Creating a Dataframe for Perplexity=50
tsne_tfidf_df_50 = pd.DataFrame(data=tsne_tfidf_50, columns=["x50", "y50"])
tsne_tfidf_df_50["target"] = df_combined["target"].values

plt.figure(figsize=(14,8))
plt.scatter(tsne_tfidf_df_50.x50, 
            tsne_tfidf_df_50.y50, 
            alpha=1,
            c=tsne_tfidf_df_50.target,
            cmap=plt.cm.coolwarm)
plt.title("T-SNE plot in SVD space (perplexity=50)")
plt.legend()
plt.show()
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

text_list = list(df_combined['Text'])
documents = [TaggedDocument(doc, tags = [str(i)]) for i, doc in enumerate(text_list)]

print ("First text List\n", text_list[0])
print ("\nTagged Document corresponding to the same\n", documents[0])
max_epochs = 100
alpha = 0.025
model = Doc2Vec(documents,
               size = 10,
               min_alpha = 0.00025,
               alpha = alpha,
               min_count = 1,
               workers = 4)
tsne_model = TSNE(n_components=2,
                  verbose=1,
                  random_state=1999,
                  n_iter=10000,
                  learning_rate = 100,
                  perplexity = 50,
                 )
tsne_d2v = tsne_model.fit_transform(model.docvecs.vectors_docs)
tsne_d2v_df = pd.DataFrame(tsne_d2v, columns = ['x', 'y'])
tsne_d2v_df['text'] = df_combined['Text']
tsne_d2v_df['target'] = df_combined['target']
tsne_d2v_df = tsne_d2v_df.dropna()
tsne_d2v_df.head()
output_notebook()
plot_d2v = bp.figure(plot_width = 800, plot_height = 700, 
                       title = "T-SNE applied to Doc2vec document embeddings",
                       tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                       x_axis_type = None, y_axis_type = None, min_border = 1)

# colormap = np.array(["#6d8dca", "#d07d3c"])
colormap = np.array(["orange", "red"])

# palette = d3["Category10"][len(tsne_tfidf_df["asset_name"].unique())]
source = ColumnDataSource(data = dict(x = tsne_d2v_df["x"], 
                                      y = tsne_d2v_df["y"],
                                      color = colormap[tsne_d2v_df["target"].astype(int) - 1],
                                      author = df_combined['Author'],
                                      sentence = tsne_d2v_df["text"],
                                      target = tsne_d2v_df["target"]))

plot_d2v.scatter(x = "x", 
                   y = "y", 
                   color="color",
                   legend = "target",
                   source = source,
                   alpha = 0.7)
hover = plot_d2v.select(dict(type = HoverTool))
hover.tooltips = {"sentence": "@sentence", 
                  "target":"@target",
                  'author' : '@author'
                 }

show(plot_d2v)

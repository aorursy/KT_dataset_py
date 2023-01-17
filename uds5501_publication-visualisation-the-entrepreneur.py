# Importing the relevant libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer 
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from string import punctuation

import re
from functools import reduce

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

df = pd.read_csv('../input/output.csv')
df = df.dropna()

df['Author'].value_counts()
# df['Author'].unique()
# df[df.Author == 'Sarah A. Downey']
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
df['Text'] = df['Text'].apply(lambda x : x.lower())
df['Text'] = df['Text'].apply(lambda x : remove_punctuation(x))
df['Text'] = df['Text'].apply(lambda x : word_tokenize(x))
df['Text'] = df['Text'].apply(lambda x : remove_stopwords(x))
df['Text'] = df['Text'].apply(lambda x : lemmatize_text(x))
def Labels(x):
    if x == 'Deb Knobelman, PhD':
        return 1
    elif x == 'Jordan “J” Gross':
        return 2
    elif x == 'Dave Schools':
        return 3
    else :
        return 4
df['labels'] = df['Author'].apply(Labels)
tf_idf_vec = TfidfVectorizer(min_df = 3,
                            max_features = 100000,
                            analyzer = 'word',
                            ngram_range = (1, 2),
                            stop_words = 'english')
tf_idf = tf_idf_vec.fit_transform(list(df['Text'].map(lambda tokens : ' '.join(tokens))))

tf_idf
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 50, random_state = 1999)
svd_tfidf = svd.fit_transform(tf_idf)
print("Dimensionality of LSA space: {}".format(svd_tfidf.shape))
# svd_tfidf[:,0]
df.labels.values
# Showing scatter plots 
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(16,12))

ax = Axes3D(fig)
ax.scatter(svd_tfidf[:,0],
          svd_tfidf[:,1],
          svd_tfidf[:,2],
          c = df.labels.values,
          cmap = plt.cm.RdBu,
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
# from MulticoreTSNE import MulticoreTSNE as TSNE

# tsne_model = TSNE(n_jobs=4,
#                   n_components=4,
#                   verbose=1,
#                   random_state=1999,
#                   n_iter=500)
# tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components = 2,
                  verbose = 1,
                  random_state = 1999,
                  n_iter = 500)
tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
tsne_tfidf_df = pd.DataFrame(data=tsne_tfidf, columns=["x", "y"])
# tsne_tfidf_df["qid"] = train_rebal["qid"].values
tsne_tfidf_df["text"] = df["Text"].values
tsne_tfidf_df["target"] = df['labels'].values
tsne_tfidf_df.head()
output_notebook()
plot_tfidf = bp.figure(plot_width = 800, plot_height = 700, 
                       title = "T-SNE applied to Tfidf_SVD space",
                       tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                       x_axis_type = None, y_axis_type = None, min_border = 1)

# colormap = np.array(["#6d8dca", "#d07d3c"])
colormap = np.array(['red', 'orange', 'green', 'darkblue'])

# palette = d3["Category10"][len(tsne_tfidf_df["asset_name"].unique())]
source = ColumnDataSource(data = dict(x = tsne_tfidf_df["x"], 
                                      y = tsne_tfidf_df["y"],
                                      color = colormap[tsne_tfidf_df["target"] - 1],
                                      text = tsne_tfidf_df["text"],
                                      target = tsne_tfidf_df["target"]))

plot_tfidf.scatter(x = "x", 
                   y = "y", 
                   color="color",
                   legend = "target",
                   source = source,
                   alpha = 0.7)

hover = plot_tfidf.select(dict(type = HoverTool))
hover.tooltips = {"text": "@text", 
                  "target":"@target"}

show(plot_tfidf)
# Perplexity = 5
tsne_model_5 = TSNE(perplexity=5,
                  n_components=2,
                  verbose=1,
                  random_state=1999,
                  n_iter=500)

tsne_tfidf_5 = tsne_model_5.fit_transform(svd_tfidf)
# Creating a Dataframe for Perplexity=5
tsne_tfidf_df_5 = pd.DataFrame(data=tsne_tfidf_5, columns=["x5", "y5"])
tsne_tfidf_df_5["target"] = df["labels"].values

plt.figure(figsize=(14,8))
plt.scatter(tsne_tfidf_df_5.x5, 
            tsne_tfidf_df_5.y5, 
            alpha=1,
            c=tsne_tfidf_df_5.target,
            cmap=plt.cm.coolwarm)
plt.title("T-SNE plot in SVD space (perplexity=5)")
plt.legend()
plt.show()

# Perplexity = 25
tsne_model_25 = TSNE(perplexity=25,
                  n_components=2,
                  verbose=1,
                  random_state=1999,
                  n_iter=500)

tsne_tfidf_25 = tsne_model_25.fit_transform(svd_tfidf)
# Creating a Dataframe for Perplexity=25
tsne_tfidf_df_25 = pd.DataFrame(data=tsne_tfidf_25, columns=["x25", "y25"])
tsne_tfidf_df_25["target"] = df["labels"].values

plt.figure(figsize=(14,8))
plt.scatter(tsne_tfidf_df_25.x25, 
            tsne_tfidf_df_25.y25, 
            alpha=1,
            c=tsne_tfidf_df_25.target,
            cmap=plt.cm.coolwarm)
plt.title("T-SNE plot in SVD space (perplexity=25)")
plt.legend()
plt.show()

# Perplexity = 50
tsne_model_50 = TSNE(perplexity=50,
                  n_components=2,
                  verbose=1,
                  random_state=1999,
                  n_iter=500)

tsne_tfidf_50 = tsne_model_50.fit_transform(svd_tfidf)
# Creating a Dataframe for Perplexity=25
tsne_tfidf_df_50 = pd.DataFrame(data=tsne_tfidf_50, columns=["x50", "y50"])
tsne_tfidf_df_50["target"] = df["labels"].values

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
text_list = list(df['Text'])
# Creating a list of terms and a list of labels to go with it
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
                  n_iter=300)
tsne_d2v = tsne_model.fit_transform(model.docvecs.vectors_docs)
tsne_d2v_df = pd.DataFrame(tsne_d2v, columns = ['x', 'y'])
tsne_d2v_df['text'] = df['Text']
tsne_d2v_df['target'] = df['labels']
tsne_d2v_df = tsne_d2v_df.dropna()
tsne_d2v_df.head()
output_notebook()
plot_d2v = bp.figure(plot_width = 800, plot_height = 700, 
                       title = "T-SNE applied to Doc2vec document embeddings",
                       tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                       x_axis_type = None, y_axis_type = None, min_border = 1)

# colormap = np.array(["#6d8dca", "#d07d3c"])
colormap = np.array(["darkblue", "cyan", "red", "orange"])

# palette = d3["Category10"][len(tsne_tfidf_df["asset_name"].unique())]
source = ColumnDataSource(data = dict(x = tsne_d2v_df["x"], 
                                      y = tsne_d2v_df["y"],
                                      color = colormap[tsne_d2v_df["target"].astype(int) - 1],
                                      question_text = tsne_d2v_df["text"],
                                      target = tsne_d2v_df["target"]))

plot_d2v.scatter(x = "x", 
                   y = "y", 
                   color="color",
                   legend = "target",
                   source = source,
                   alpha = 0.7)
hover = plot_d2v.select(dict(type = HoverTool))
hover.tooltips = {"question_text": "@question_text", 
                  "target":"@target"}

show(plot_d2v)
type(tsne_d2v_df['target'][0])

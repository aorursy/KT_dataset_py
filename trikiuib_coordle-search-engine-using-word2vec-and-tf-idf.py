# Install required pip packages. Please note that the internet must be switched on to install them in the Kaggle kernel.
!pip install -U kneed

# Imports
import os
from os.path import join as join_path
import numpy as np
rng_seed = 368
np.random.seed(rng_seed)
import pandas as pd

import warnings
from numba.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning) # Silence NumbaPerformanceWarning for UMAP
import umap
from sklearn.cluster import KMeans

from kneed import KneeLocator
from scipy.spatial.distance import pdist
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import plotly.express as px

from IPython.display import IFrame
IFrame('https://coordle.triki.no/', width=800, height=600)
!pip install -U https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
!pip install -U git+https://github.com/JonasTriki/inf368-exercise-3-coordle.git
    
# Import Coordle modules
from coordle.preprocessing import CORD19Data
from coordle.utils import clean_text
from coordle.backend import QueryAppenderIndex
# Define some constants
kaggle_input_dir = join_path('/', 'kaggle', 'input')
cord_data_raw_dir = join_path(kaggle_input_dir, 'CORD-19-research-challenge')
# Perform preprocessing on the raw data
cord_df = CORD19Data(cord_data_raw_dir).process_data()
#  Sanity check the processed dataframe
cord_df.head()
# Implement the data interator for Word2Vec
class CORDDataIteratorWord2Vec():
    def __init__(self, texts: np.ndarray):
        self.texts = texts
    
    def __iter__(self):
        for text in self.texts:
            sentences = nltk.tokenize.sent_tokenize(text)
            cleaned_sentences = [clean_text(sent) for sent in sentences]
            for sentence in cleaned_sentences:
                yield sentence
# Implement the epoch saver for Word2Vec
class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, output_dir: str, prefix: str, start_epoch: int = 1):
        self.output_dir = output_dir
        self.prefix = prefix
        self.epoch = start_epoch

    def on_epoch_end(self, model):
        output_path = join_path(self.output_dir, f'{self.prefix}_epoch_{self.epoch}.model')
        model.save(output_path)
        self.epoch += 1
# Load the trained Gensim model
model_path = join_path(kaggle_input_dir, 'gensim-word2vec-model', 'cord-19-w2v.model')
w2v_model = Word2Vec.load(model_path)
word_embedding_matrix = w2v_model.trainables.syn1neg
w2v_model.wv.most_similar('covid')
w2v_model.wv.most_similar('virus')
w2v_model.wv.most_similar('pandemic')
# Cluster
min_k = 2
ks = np.arange(min_k, 21)
errors = np.zeros(len(ks))
clusterings = np.zeros((len(ks), word_embedding_matrix.shape[0]))
for k in ks:
    print(f'Clustering using k={k}...')
    clusterer = KMeans(n_clusters=k, n_jobs=-1)
    pred_labels = clusterer.fit_predict(word_embedding_matrix)
    clusterings[k - min_k] = pred_labels
    errors[k - min_k] = clusterer.inertia_
# Show the elbow plot to determine the best k
kneedle = KneeLocator(ks, errors, S=1.0, curve='convex', direction='decreasing')
kneedle.plot_knee()

# Select best clustering
best_clustering = clusterings[kneedle.knee - min_k]
# Reduce dimensionality using UMAP (with default params)
word_embedding_3d = umap.UMAP(n_components=3).fit_transform(word_embedding_matrix)
# Visualize the words in 3D with Plotly
word_embedding_vis_df = pd.DataFrame({
    'x': word_embedding_3d[:, 0],
    'y': word_embedding_3d[:, 1],
    'z': word_embedding_3d[:, 2],
    'cluster_label': best_clustering,
    'word': w2v_model.wv.index2word
})
fig = px.scatter_3d(word_embedding_vis_df, x='x', y='y', z='z', color='cluster_label', hover_name='word')
fig.show()
# To demonstrate how the search engine works, we index on a subset of the documents in the CORD-19 dataframe.
ai_index = QueryAppenderIndex(w2v_model.wv.most_similar, n_similars=1)
ai_index.build_from_df(
    cord_df[:1000],
    'cord_uid',
    'title',
    'body_text', 
    verbose=True, 
    use_multiprocessing=True,
    workers=-1
)
def search_and_show(query: str, max_results: int = 3, max_body_length: int = 500):
    '''Searches using the AI Index and shows the result
    
    Args:
        query: Search query
        max_results: Max results to show for each query    
    '''
    docs, scores, errmsgs = ai_index.search(query)
    if errmsgs:
        print('The following errors occurred:', errmsgs)
    else:
        if len(docs) == 0:
            print('Sorry, no results found.')
        else:
            for doc, score in zip(docs[:max_results], scores[:max_results]):
                print(f'{doc.uid}  {str(doc.title)[:70]:<70}  {score:.4f}')
                print('---')
                print(f'{cord_df[cord_df.cord_uid == doc.uid].body_text.values[0][:max_body_length]} {...}')
                print('---')
search_and_show('virus')
search_and_show('virus AND')
search_and_show('coronavirus symptoms in humans')
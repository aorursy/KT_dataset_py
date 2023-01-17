import gc

import numpy as np

import pandas as pd

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfTransformer

from pathlib import Path



INPUT_PATH = Path("../input/otto-group-product-classification-challenge/")
train = pd.read_csv(INPUT_PATH / "train.csv")

test = pd.read_csv(INPUT_PATH / "test.csv")



train.shape, test.shape
drop_cols = ["id"]

target_col = "target"

target = train[target_col]

feat_cols = [col for col in train.columns if col not in drop_cols + [target_col]]

 

train.drop(columns=drop_cols + [target_col], inplace=True)

test.drop(columns=drop_cols, inplace=True)
def to_tfidf(df, feat_cols, csv_path):

    transformer = TfidfTransformer()

    _tfidf = pd.DataFrame(transformer.fit_transform(df).toarray())

    _tfidf.columns = feat_cols

    _tfidf.to_csv(csv_path, index=False)
to_tfidf(train, feat_cols, "train_tfidf.csv")

to_tfidf(test, feat_cols, "test_tfidf.csv")
!pip install -q opentsne
from openTSNE import TSNE

from openTSNE.callbacks import ErrorLogger

n_dim = 2

tsne = TSNE(n_components=n_dim, perplexity=50, n_iter=500, random_state=42, n_jobs=8, callbacks=ErrorLogger())
data = pd.concat([train, test]).reset_index()

embedding = tsne.fit(data)
embedding = pd.DataFrame(embedding)

embedding.columns = ["dim1", "dim2"]

embedding.to_csv("tsne_2dim.csv", index=False)
train_tfidf = pd.read_csv("train_tfidf.csv")

test_tfidf = pd.read_csv("test_tfidf.csv")



train_tfidf.shape, test_tfidf.shape
data = pd.concat([train_tfidf, test_tfidf]).reset_index()

embedding = tsne.fit(data)
embedding = pd.DataFrame(embedding)

embedding.columns = ["dim1", "dim2"]

embedding.to_csv("tfidf_tsne_2dim.csv", index=False)
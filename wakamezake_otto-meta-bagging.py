!pip install -q opentsne
import gc

import numpy as np

import pandas as pd

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
data = pd.concat([train, test])

data.shape
data = data.apply(lambda x: 1 / (1 + np.exp(-np.sqrt(x))))
data.head()
from openTSNE import TSNE

from openTSNE.callbacks import ErrorLogger

n_dim = 2

# ref

# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

# https://www.rdocumentation.org/packages/Rtsne/versions/0.15/topics/Rtsne

tsne = TSNE(n_components=n_dim, perplexity=30,

            n_iter=1000, random_state=42, n_jobs=8, callbacks=ErrorLogger())
embedding = tsne.fit(data.values)
embedding = pd.DataFrame(embedding)

embedding.columns = ["dim1", "dim2"]

embedding.to_csv("tsne_2dim.csv", index=False)
import matplotlib.pyplot as plt

import seaborn as sns



tr_embedding = embedding[:train.shape[0]]

tr_embedding[target_col] = target



plt.figure(figsize=(20, 10))

sns.scatterplot(x="dim1", y="dim2", data=tr_embedding, hue=target_col, size=9)
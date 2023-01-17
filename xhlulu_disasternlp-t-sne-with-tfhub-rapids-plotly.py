%%time

import sys

!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cuml

from cuml.manifold import TSNE, UMAP

import plotly.express as px

import pandas as pd

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
%%time

module_url = '/kaggle/input/universalsentenceencoderlarge4/'

embed = hub.KerasLayer(module_url, trainable=True, name='USE_embedding')
%%time

encodings = embed(train.text)['outputs'].numpy()
%%time

tsne2d = TSNE(n_components=2)

projections_2d = tsne2d.fit_transform(encodings)
%%time

umap3d = UMAP(n_components=3)

projections_3d = umap3d.fit_transform(encodings)
labels = train.target.apply(lambda x: 'Real Disaster' if x else 'Not Disaster')
fig = px.scatter(

    x=projections_2d[:, 0], y=projections_2d[:, 1], 

    color=labels, hover_name=train.text, height=700

)

fig.show()
fig = px.scatter_3d(

    x=projections_3d[:, 0], y=projections_3d[:, 1], z=projections_3d[:, 2], 

    color=labels, hover_name=train.text, size_max=2, height=700

)

fig.update_traces(marker=dict(size=3))



fig.show()
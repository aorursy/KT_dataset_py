import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
df_doc_meta = pd.read_csv('../input/documents_meta.csv')

df_doc_categories = pd.read_csv('../input/documents_categories.csv')

df_doc_entities = pd.read_csv('../input/documents_entities.csv')

df_doc_topics = pd.read_csv('../input/documents_topics.csv')
df_doc_meta.head()
df_doc_categories.head()
category_count = df_doc_categories.groupby('category_id').category_id.count()

plt.hist(category_count, bins=50)
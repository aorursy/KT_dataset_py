import numpy as np

import pandas as pd

from sklearn.metrics import pairwise

import tensorflow as tf

import tensorflow_datasets as tfds

from IPython.display import display
from google.cloud import bigquery

client = bigquery.Client()
# All of the standard patent data is available in bigquery such as publication/

# grant/filing/priority date, inventors, assignees, cpc/ipc/etc codes, citations



df = client.query("""

SELECT

  publication_number, country_code, publication_date, filing_date, 

  priority_date, grant_date, inventor, assignee, cpc, citation, priority_claim

FROM

  `patents-public-data.patents.publications` 

WHERE

  RAND() < 0.01 

LIMIT

  5""").to_dataframe()
df
# Some of the fields are nested and can be either unnested in bigquery or

# python. Here we unnest all of the cpc and citation data. The unnesting of two

# fields creates a cross join between them, so even for a single patent a lot

# of rows are created.



df = client.query("""



SELECT 

  pubs.publication_number, 

  pubs.filing_date,

  cpc.code as cpc_code, 

  cpc.first as cpc_first,

  cpc.inventive as cpc_inventive,

  cite.publication_number AS cite_pub,

  cite.filing_date AS cite_filing_date

FROM 

  `patents-public-data.patents.publications` AS pubs,

  UNNEST(citation) AS cite,

  UNNEST(cpc) AS cpc

WHERE 

  pubs.publication_number = "US-8000000-B2"

  """).to_dataframe()

df.head(20)
 # There is a lot of more detailed patent data that can be obtained,

 # such as the full text of the patent (title, abstract, claims, description),

 # its top terms and a 64 dimension embedding representation.

 

df = client.query("""



SELECT 

  pubs.publication_number,

  abstract.text AS abstract,

  claims.text AS claim,

  top_terms,

  embedding_v1 AS embedding

FROM 

  `patents-public-data.patents.publications` AS pubs

    INNER JOIN `patents-public-data.google_patents_research.publications` AS res ON 

      pubs.publication_number = res.publication_number,

    UNNEST(abstract_localized) AS abstract,

    UNNEST(claims_localized) AS claims

WHERE 

  pubs.publication_number = "US-8000000-B2"

  """).to_dataframe()
df
# Get Embeddings for a random set of patents



df = client.query("""

SELECT 

  publication_number,

  embedding_v1 as embedding

FROM 

  `patents-public-data.google_patents_research.publications`

WHERE 

  country = "United States"

  AND RAND() < 0.1

LIMIT

  100""").to_dataframe()
# Compute similarity between the patents.

emb = df.embedding.to_list()

pairwise.cosine_similarity(emb, emb)
# We can cluster the embeddings too

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=0).fit(emb)

kmeans.labels_
# We fetch the embeddings for 10000 random US patents and use the first

# letter of the cpc code as the classification for a patent.



df = client.query("""



SELECT 

  publication_number,

  embedding_v1,

  SUBSTR(cpc.code, 0, 1) AS cpc_class

FROM 

  `patents-public-data.google_patents_research.publications`,

  UNNEST(cpc) AS cpc

WHERE 

  country = "United States"

  AND RAND() < 0.1

LIMIT

  10000""").to_dataframe()
cpc_classes = {

    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'Y': 8}

classes = tf.convert_to_tensor([cpc_classes[x] for x in df.cpc_class.tolist()])



inputs = tf.convert_to_tensor(df.embedding_v1.tolist())
model = tf.keras.Sequential([

  tf.keras.layers.Dense(32, activation='relu'),

  tf.keras.layers.Dense(len(cpc_classes))

])



model.compile(

    optimizer='adam',

    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

    metrics=['accuracy']

)



history = model.fit(x=inputs,  y=classes, epochs=10, validation_split=0.1)
# For another set of 10000 random US patents we fetch the abstract text and

# use the first letter of the cpc code for the classification.



df = client.query("""



SELECT 

  publication_number,

  abstract,

  SUBSTR(cpc.code, 0, 1) AS cpc_class

FROM 

  `patents-public-data.google_patents_research.publications`,

  UNNEST(cpc) AS cpc

WHERE 

  country = "United States"

  AND RAND() < 0.1

LIMIT

  100000""").to_dataframe()
cpc_classes = {

    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'Y': 8}

classes = tf.convert_to_tensor([cpc_classes[x] for x in df.cpc_class.tolist()])



vocab_size = 1000

emb_size = 16

tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(

    df.abstract.tolist(), vocab_size)
msl = 100  # max sequence length

inputs = [tokenizer.encode(x) for x in df.abstract.tolist()]

inputs = [x[:msl] if len(x) > msl else x + [0]*(msl-len(x)) for x in inputs]

inputs = tf.convert_to_tensor(inputs)
model = tf.keras.Sequential([

  tf.keras.layers.Embedding(vocab_size, emb_size),

  tf.keras.layers.Conv1D(16, 3, padding='same', activation='relu'),

  tf.keras.layers.MaxPooling1D(),

  tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),

  tf.keras.layers.GlobalAveragePooling1D(),

  tf.keras.layers.Dense(32, activation='relu'),

  tf.keras.layers.Dense(len(cpc_classes))

])



model.compile(

    optimizer='adam',

    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

    metrics=['accuracy']

)



history = model.fit(x=inputs, y=classes, epochs=10, validation_split=0.1)
!pip install sister
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

from gensim.models import Word2Vec

import numpy as np

from sklearn.model_selection import train_test_split

import sister



# You would not need these since they are used to hack kaggle memory limitation

from sister.word_embedders import FasttextEmbedding

from fasttext import load_model
# embedder = sister.MeanEmbedding(lang="en")
class workaround():

    pass



path = '/kaggle/input/sister/wiki.simple.bin'

fasttext_model = load_model(path)



model = workaround()

model.__class__ = FasttextEmbedding



model.model = fasttext_model

embedder = sister.MeanEmbedding(lang="en", word_embedder=model)
sentence = "I love kaggle."

vector = embedder(sentence)  # 300-dim vector



print(vector)
data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

data = data.drop([6771]) # There is missing data in this row so we just remove it.
data['text_embedding'] = data.apply(lambda row: embedder(row.text), axis=1)

data['keyword_embedding'] = data.apply(lambda row: embedder(row.keyword) if not row.keyword is np.nan else np.zeros(300), axis=1)

data['location_embedding'] = data.apply(lambda row: embedder(row.location) if not row.location is np.nan else np.zeros(300), axis=1)
X_df = data[['text_embedding', 'keyword_embedding', 'location_embedding']]

X = X_df.apply(lambda row: np.concatenate([row.text_embedding, row.keyword_embedding, row.location_embedding]), axis=1)

X = X.values

X = np.stack(X, axis=0)

y = data['target']

y = y.values.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.optimizers import Adam

from keras.losses import mean_squared_logarithmic_error, mean_squared_error, binary_crossentropy



model = Sequential([

    Dense(128, input_shape=(900,)),

    Activation('relu'),

    Dense(256, input_shape=(900,)),

    Activation('relu'),

    Dense(128, input_shape=(900,)),

    Activation('relu'),

    Dense(128, input_shape=(900,)),

    Activation('relu'),

    Dense(1),

    Activation('sigmoid'),

])



model.compile(optimizer=Adam(learning_rate=0.0001), loss=binary_crossentropy)
model.fit(X_train, y_train, epochs=100, batch_size=32)
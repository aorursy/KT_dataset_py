# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline  

import time
start_time = time.time()
from sklearn.model_selection import train_test_split
import sys, os, re, csv, codecs, numpy as np, pandas as pd
np.random.seed(32)
os.environ["OMP_NUM_THREADS"] = "4"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D

import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
tweets = pd.read_csv("../input/clinton-trump-tweets/tweets.csv")
embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
tweets.head()
tweets.shape
embed_size = 300
max_features = 130000
max_len = 220

tweets["text"].fillna("no comment")
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
raw_text = tweets["text"].str.lower()

tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(raw_text)
tweets["comment_seq"] = tk.texts_to_sequences(raw_text)
tweets_pad_sequences = pad_sequences(tweets.comment_seq, maxlen = max_len)
tweets_pad_sequences.shape
tweets_pad_sequences
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
model = load_model("../input/bi-gru-lstm-cnn-poolings-fasttext/best_model.hdf5")
pred = model.predict(tweets_pad_sequences, batch_size = 1024, verbose = 1)
pred.max()
toxic_predictions = pd.DataFrame(columns=list_classes, data=pred)
toxic_predictions.head()
toxic_predictions['id'] = tweets['id'].values
toxic_predictions['handle'] = tweets['handle'].values
toxic_predictions['text'] = tweets['text'].values
toxic_predictions.tail()
Hillary_predictions = toxic_predictions[toxic_predictions['handle'] == 'HillaryClinton']
Trump_predictions = toxic_predictions[toxic_predictions['handle'] == 'realDonaldTrump']
Hillary_predictions[list_classes].describe()
Trump_predictions[list_classes].describe()
melt_df = pd.melt(toxic_predictions, value_vars=list_classes, id_vars='handle')
melt_df.head()
sns.violinplot(x='variable', y='value', hue='handle', data=melt_df)
plt.show()
melt_df['value'] = np.clip(melt_df['value'].values, 0, 0.2)
sns.violinplot(x='variable', y='value', hue='handle', data=melt_df)
plt.show()
melt_df['value'] = np.clip(melt_df['value'].values, 0, 0.05)
sns.violinplot(x='variable', y='value', hue='handle', data=melt_df)
plt.show()
Hillary_predictions.loc[Hillary_predictions['toxic'].idxmax()]['text']
print(Hillary_predictions.sort_values(by=['toxic'], ascending=False)['text'].head(10).values)
print(Hillary_predictions.sort_values(by=['severe_toxic'], ascending=False)['text'].head(10).values)
print(Hillary_predictions.sort_values(by=['obscene'], ascending=False)['text'].head(10).values)
Hillary_predictions.loc[Hillary_predictions['threat'].idxmax()]['text']
print(Hillary_predictions.sort_values(by=['threat'], ascending=False)['text'].head(10).values)

Hillary_predictions.loc[Hillary_predictions['insult'].idxmax()]['text']
print(Hillary_predictions.sort_values(by=['insult'], ascending=False)['text'].head(10).values)
Hillary_predictions.loc[Hillary_predictions['identity_hate'].idxmax()]['text']

print(Hillary_predictions.sort_values(by=['identity_hate'], ascending=False)['text'].head(10).values)
Trump_predictions.loc[Trump_predictions['toxic'].idxmax()]['text']
print(Trump_predictions.sort_values(by=['toxic'], ascending=False)['text'].head(10).values)
print(Trump_predictions.sort_values(by=['severe_toxic'], ascending=False)['text'].head(10).values)
print(Trump_predictions.sort_values(by=['obscene'], ascending=False)['text'].head(10).values)
Trump_predictions.loc[Trump_predictions['threat'].idxmax()]['text']
print(Trump_predictions.sort_values(by=['threat'], ascending=False)['text'].head(10).values)
Trump_predictions.loc[Trump_predictions['insult'].idxmax()]['text']
print(Trump_predictions.sort_values(by=['insult'], ascending=False)['text'].head(10).values)
Trump_predictions.loc[Trump_predictions['identity_hate'].idxmax()]['text']
print(Trump_predictions.sort_values(by=['identity_hate'], ascending=False)['text'].head(10).values)


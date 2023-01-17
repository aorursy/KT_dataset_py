# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import re
import seaborn as sns


sns.set(style='darkgrid',
              font_scale=1.5,
              rc={'figure.figsize': (12, 6)})
data_path = Path('/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv')
df = pd.read_csv(data_path)
df.info()
df.head()
df['Y'].value_counts()/df['Y'].size
# Extract features by on target class
def get_class_feature(df, cls, feature):
    """
    Extract featuers by class
    Returns new dataframe with respective class
    """
    return df.loc[df['Y'] == cls, feature]
def extract_title_len(df):
    """
    Returns title length as a pandas series
    """
    return df['Title'].apply(len)
df['title_len'] = extract_title_len(df)
lq_close = get_class_feature(df, 'LQ_CLOSE', 'title_len')
lq_edit = get_class_feature(df, 'LQ_EDIT', 'title_len')
hq = get_class_feature(df, 'HQ', 'title_len')
sns.distplot(lq_close, label='LQ Close Title Length')
sns.distplot(lq_edit, label='LQ Edit Title Length')
sns.distplot(hq, label='HQ Title Length')
plt.title('Distribution of Title Length')
plt.ylabel('Density')
plt.xlabel('Title Length')
plt.legend();
def extract_body_len(df):
    """
    Returns body length as a pandas series
    """
    return df['Body'].apply(len)
df['body_len'] = extract_body_len(df)
lq_close = get_class_feature(df, 'LQ_CLOSE', 'body_len')
lq_edit = get_class_feature(df, 'LQ_EDIT', 'body_len')
hq = get_class_feature(df, 'HQ', 'body_len')
sns.distplot(lq_close, label='LQ Close Body Length')
sns.distplot(lq_edit, label='LQ Edit Body Length')
sns.distplot(hq, label='HQ Body Length')
plt.title('Distribution of Body Length')
plt.ylabel('Density')
plt.xlabel('Body Length')
plt.legend();
def extract_tags_len(df):
    """
    Returns tags length as a pandas series
    """
    return df['Tags'].apply(lambda tag: len(re.findall('<([^>]*)>', tag)))
df['tags_len'] = extract_tags_len(df)
lq_close = get_class_feature(df, 'LQ_CLOSE', 'tags_len')
lq_edit = get_class_feature(df, 'LQ_EDIT', 'tags_len')
hq = get_class_feature(df, 'HQ', 'tags_len')
sns.distplot(lq_close, label='LQ Close Tags Length')
sns.distplot(lq_edit, label='LQ Edit Tags Length')
sns.distplot(hq, label='HQ Tags Length')
plt.title('Distribution of Tags Length')
plt.ylabel('Density')
plt.xlabel('Tags Length')
plt.legend();
df['tags_exploded'] = df['Tags'].apply(lambda tags: re.findall('<([^>]*)>', tags))
tags_count = Counter()

for tags in df['tags_exploded']:
    tags_count.update(tags)
tags_count = pd.DataFrame.from_dict(tags_count, orient='index').reset_index()
tags_count = tags_count.rename(columns={'index': 'tag', 0: 'count'})
coverage = tags_count.sort_values(by='count',ascending=False).head(50)
plt.figure(figsize=(12, 20))
sns.barplot(y='tag', x='count', data=coverage)
plt.xlabel('Tag Counts')
plt.ylabel('Tags')
plt.title('Top 50 tags frequency');
# Most common tags are contained in 1% or more posts
mask = (tags_count['count'].sort_values(ascending=False)/df.shape[0]) > 0.009
common_tags = (tags_count[mask]['tag'].tolist())
def filter_tag(row):
    """
    Returns list of tags not in common tags
    """
    return list(filter(lambda tag: tag in common_tags, row))


df['filter_tag'] = df['tags_exploded'].apply(filter_tag)
def extract_filter_tag_len(df):
    """
    Returns length of filter tags as a pandas series
    """
    return df['filter_tag'].apply(len)
df['filter_tags_len'] = extract_filter_tag_len(df)
lq_close = get_class_feature(df, 'LQ_CLOSE', 'filter_tags_len')
lq_edit = get_class_feature(df, 'LQ_EDIT', 'filter_tags_len')
hq = get_class_feature(df, 'HQ', 'filter_tags_len')
sns.distplot(lq_close, label='LQ Close Tags Length')
sns.distplot(lq_edit, label='LQ Edit Tags Length')
sns.distplot(hq, label='HQ Tags Length')
plt.title('Distribution of Filtered Tags Length')
plt.ylabel('Density')
plt.xlabel('Filtered Tags Length')
plt.legend();
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Seeding
seed = 42
random.seed(seed)
np.random.seed(seed)
df['text'] = df['Title'] + ' ' + df['Body']
data = df[['text', 'Y']]
x = data['text']
y = data['Y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
def clean_text(df):
    """
    Removes special escape characters and html tags
    Returns dataframe without escape characters or html tags
    """
    return df.str.replace('[\r\t\n]|<([^>]*)>', ' ')

clean_transformer = FunctionTransformer(lambda df: clean_text(df))
# html_transformer,
#                      endline_transformer,
                     

pipe = make_pipeline(CountVectorizer(max_features=10000),
                     TfidfTransformer())

le = LabelEncoder()
y_train = le.fit_transform(y_train)


xx_train = pipe.fit_transform(x_train)
rf = RandomForestClassifier()
log = LogisticRegression()
np.mean(cross_val_score(log, xx_train, y_train, n_jobs=-1))
np.mean(cross_val_score(rf, xx_train, y_train, n_jobs=-1))
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
from time import time
rf = clone(pipe)
log = clone(pipe)
parameters = {
    'countvectorizer__max_df': (0.5, 0.75, 1.0),
    'countvectorizer__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
    'tfidftransformer__norm': ('l1', 'l2')
}
rf.steps.append(('rf', RandomForestClassifier()))
log.steps.append(('log', LogisticRegression(max_iter=100000000000000)))
t0 = time()
search = RandomizedSearchCV(log, parameters, n_jobs=-1, verbose=True)
search.fit(x_train, y_train)
t1 = time()
print(search.best_score_)
print(search.best_params_)
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
pred = search.best_estimator_.predict(x_test)
y_test = le.transform(y_test)
accuracy_score(y_test, pred)
plot_confusion_matrix(search.best_estimator_, 
                      x_test, 
                      y_test,
                      labels=[0, 1, 2],
                      display_labels=le.classes_)

plt.grid(False)
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import text, sequence
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Dropout, Bidirectional, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
seed = 42 
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
df['text'] = df['Title'] + ' ' + df['Body']
data = df[['text', 'Y']]

x = data['text']
y = data['Y']
def clean_text(df):
    """
    Removes special escape characters and html tags
    Returns dataframe without escape characters or html tags
    """
    return df.str.replace('[\r\t\n]|<([^>]*)>', ' ')
le = LabelEncoder()
y = le.fit_transform(y)

# x = clean_text(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)

max_features = 10000
max_len = 300
tokenizer = text.Tokenizer(num_words=max_len)
tokenizer.fit_on_texts(x_train)
def text_to_seq(tokenizer, df, maxlen):
    """
    Converts text to sequence of tokens
    Returns NumPy array of max length
    """
    tokenized_data = tokenizer.texts_to_sequences(df)
    return sequence.pad_sequences(tokenized_data, 
                                  maxlen=maxlen, 
                                  truncating='post')
x_train = text_to_seq(tokenizer, x_train, max_len)
x_test = text_to_seq(tokenizer, x_test, max_len)
glove_path = Path('/kaggle/input/glove-global-vectors-for-word-representation/')
embedding_path = glove_path.joinpath('glove.6B.200d.txt')
# /kaggle/input/glove-global-vectors-for-word-representation/glove.6B.200d.txt
# /kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt
def get_word_embed(line):
    """
    Parse word and respective embedding by line from glove file
    Returns word and embedding as tuple
    """
    word, *embedding = line.split(' ')
    return word, np.asarray(embedding, dtype=np.float32)
with open(embedding_path) as f:
    words_embed = {}
    glove_embeddings = []
    
    for line in f:
        word, embed = get_word_embed(line)
        
        words_embed[word] = embed
        glove_embeddings.append(embed)
glove_embeddings = np.asarray(glove_embeddings)
embed_mean, embed_std = glove_embeddings.mean(), glove_embeddings.std()
embed_dim = glove_embeddings.shape[1]

word2idx = tokenizer.word_index
nb_words = min(max_features, len(word2idx))

# Creating embedding matrix for our corpus
embed_matrix = np.random.normal(embed_mean, embed_std, (nb_words, embed_dim))

for word, idx in word2idx.items():
    if idx < max_features:
        embedding_vect = words_embed.get(word)
        
        if embedding_vect is not None:
            embed_matrix[idx] = embedding_vect
embed_matrix.shape
batch_size = 256
epochs = 5

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.4, min_lr=0.0000001)
tf.config.experimental.list_physical_devices()
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embed_dim, input_length=max_len, 
                        weights=[embed_matrix], trainable=True))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.4, dropout=0.4)))
    model.add(Bidirectional(LSTM(units=128, recurrent_dropout=0.2, dropout=0.2)))

    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
history = model.fit(x_train, y_train, batch_size, 
              validation_data=(x_test, y_test), epochs=epochs,
              callbacks=[learning_rate_reduction])
e = range(1, 6)

loss_result = pd.DataFrame({'epochs': e,
              'training_loss': history.history['loss'],
              'validation_loss': history.history['val_loss']
             })

sns.lineplot(data=loss_result, x='epochs', y='training_loss')
sns.lineplot(data=loss_result, x='epochs', y='validation_loss')

plt.ylabel('Loss')

plt.title('Training loss vs Validation loss');
e = range(1, 6)

loss_result = pd.DataFrame({'epochs': e,
              'training_accuracy': history.history['accuracy'],
              'validation_accuracy': history.history['val_accuracy']
             })

sns.lineplot(data=loss_result, x='epochs', y='training_accuracy' )
sns.lineplot(data=loss_result, x='epochs', y='validation_accuracy' )

plt.ylabel('accuracy')
plt.title('Training accuracy vs Validation accuracy');
import random

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import text, sequence
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Dropout, Bidirectional, Input, Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

from transformers import TFDistilBertModel
from transformers import DistilBertTokenizerFast, DistilBertConfig

df['text'] = df['Title'] + ' ' + df['Body']
data = df[['text', 'Y']]

x = data['text']
y = data['Y']

le = LabelEncoder()
y = le.fit_transform(y)

seed = 42 
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)

max_len = 300
model_name = 'distilbert-base-uncased'
model_type = TFDistilBertModel
model_tokenizer = DistilBertTokenizerFast
tokenizer = model_tokenizer.from_pretrained(model_name, lower_case=True)
sent = x_train[0]
toks = tokenizer.tokenize(sent)
ids = tokenizer.convert_tokens_to_ids(toks)

print(f'Sentence: {sent}')
print(f'Tokens: {toks}', end='\n\n')
print(f'Ids: {ids}')
def get_tokenized_inputs(data, tokenizer, max_len):
    """
    Returns tokenized data
    """
    
    input_ids = np.empty((data.shape[0], max_len))
#     attention_masks = []
    
    for idx, sent in enumerate(tqdm(data)):
        encoded_dict = (tokenizer.encode_plus(
                                sent,
                                add_special_tokens=True,
                                max_length=max_len,
                                pad_to_max_length=True,
                                return_attention_mask=False,
                                return_tensors='tf'
                        ))
        input_ids[idx, :] = encoded_dict['input_ids']
#         attention_masks.append(encoded_dict['attention_mask'])
        
    return np.squeeze(input_ids)#, np.array(attention_masks)
train_inputs = get_tokenized_inputs(x_train, tokenizer, max_len)
test_inputs = get_tokenized_inputs(x_test, tokenizer, max_len)
batch_size = 32
epochs = 5
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    transf = model_type.from_pretrained(model_name, num_labels=3)
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transf(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    output = Dense(3, activation='softmax')(cls_token)

    model = Model(inputs=input_word_ids, outputs=output)

    model.compile(optimizer=Adam(lr=7e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
history = model.fit(train_inputs, y_train, batch_size, 
              validation_data=(test_inputs, y_test), epochs=epochs)
e = range(1, 6)

loss_result = pd.DataFrame({'epochs': e,
              'training_loss': history.history['loss'],
              'validation_loss': history.history['val_loss']
             })

sns.lineplot(data=loss_result, x='epochs', y='training_loss')
sns.lineplot(data=loss_result, x='epochs', y='validation_loss')

plt.ylabel('Loss')

plt.title('Training loss vs Validation loss');
e = range(1, 6)

loss_result = pd.DataFrame({'epochs': e,
              'training_accuracy': history.history['accuracy'],
              'validation_accuracy': history.history['val_accuracy']
             })

sns.lineplot(data=loss_result, x='epochs', y='training_accuracy' )
sns.lineplot(data=loss_result, x='epochs', y='validation_accuracy' )

plt.ylabel('accuracy')
plt.title('Training accuracy vs Validation accuracy');


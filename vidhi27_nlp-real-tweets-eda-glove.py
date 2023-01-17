!pip install fse
import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud,STOPWORDS

import re

import string

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from gensim.models import KeyedVectors

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from fse.models import uSIF

from fse import IndexedList

%matplotlib inline

sns.set(style="darkgrid")
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## Load data

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.info()
test.info()
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
x = train.target.value_counts()

sns.barplot(x.index, x)
## Split training set into real disaster tweets vs not

train_real = train[train.target == 1]

train_real.target.value_counts()
train_fake = train[train.target == 0]

train_fake.target.value_counts()
fig,ax = plt.subplots(1, 2, figsize = (18,6))

ax[0].set_title('dist of #characters in disaster tweets')

ax[1].set_title('dist of #characters in non-disaster tweets')

sns.distplot(train_real.text.str.len(), ax = ax[0])

sns.distplot(train_fake.text.str.len(), ax = ax[1])

plt.setp(ax, xlim = (0, 180), ylim = (0, 0.04))

plt.show()
fig,ax = plt.subplots(1, 2, figsize = (18,6))

ax[0].set_title('dist of #words in disaster tweets')

ax[1].set_title('dist of #words in non-disaster tweets')

sns.distplot(train_real.text.str.split().apply(lambda x: len(x)), ax = ax[0])

sns.distplot(train_fake.text.str.split().apply(lambda x: len(x)), ax = ax[1])

plt.setp(ax, xlim = (0, 40), ylim = (0, 0.14))

plt.show()
fig,ax = plt.subplots(1, 2, figsize = (18,6))

ax[0].set_title('dist of average word length in disaster tweets')

ax[1].set_title('dist of average word length in non-disaster tweets')

sns.distplot(train_real.text.str.split().apply(lambda x: np.mean([len(word) for word in x])), ax = ax[0])

sns.distplot(train_fake.text.str.split().apply(lambda x: np.mean([len(word) for word in x])), ax = ax[1])

plt.setp(ax, xlim = (0, 22.5), ylim = (0, 0.35))

plt.show()
wordcloud = WordCloud(

    stopwords=set(STOPWORDS),

    background_color='white',

    scale= 3,

    random_state = 1).generate(str(train_real.text)) ## generate wordcloud

plt.figure(1, figsize=(15,10))

plt.title('most frequent words in disaster tweets')

plt.axis('off')

plt.imshow(wordcloud)
wordcloud = WordCloud(

    stopwords=set(STOPWORDS),

    background_color='white',

    scale= 3,

    random_state = 1).generate(str(train_fake.text)) ## generate wordcloud

plt.figure(1, figsize=(15,10))

plt.title('most frequent words in non-disaster tweets')

plt.axis('off')

plt.imshow(wordcloud)
def extract_hastags(df, col):

    hashtags = df[col].apply(lambda x: [match.group(0)[1:] for match in re.finditer(r"#\w+", x)])

    return pd.Series([item.lower() for lists in hashtags for item in lists if item != []]).value_counts()



hashtags_real = extract_hastags(train_real, 'text')

hashtags_fake = extract_hastags(train_fake, 'text')
fig,ax = plt.subplots(1, 2, figsize = (18,6))

ax[0].set_title('dist of top 20 hashtags in disaster tweets')

ax[1].set_title('dist of top 20 hashtags in non-disaster tweets')

sns.barplot(x = hashtags_real[:20].values, y = hashtags_real[:20].index, ax = ax[0])

sns.barplot(x = hashtags_fake[:20].values, y = hashtags_fake[:20].index, ax = ax[1])

plt.setp(ax, xlim=(0, 60))

plt.show()
train.text = train.text.apply(lambda x: x.lower())

test.text = test.text.apply(lambda x: x.lower())
def remove_url(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    return url_pattern.sub(r'', text)



train.text = train.text.apply(lambda x: remove_url(x))

test.text = test.text.apply(lambda x: remove_url(x))
def remove_html_tags(text):

    html_pattern = re.compile(r'<.*?>')

    return html_pattern.sub(r'', text)



train.text = train.text.apply(lambda x: remove_html_tags(x))

test.text = test.text.apply(lambda x: remove_html_tags(x))
def remove_emoji(text):

    emoji_pattern = re.compile("["

                               u"\U0001F600-\U0001F64F"  # emoticons

                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                               u"\U0001F680-\U0001F6FF"  # transport & map symbols

                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                               u"\U00002702-\U000027B0"

                               u"\U000024C2-\U0001F251"

                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



train.text = train.text.apply(lambda x: remove_emoji(x))

test.text = test.text.apply(lambda x: remove_emoji(x))
def remove_punctuations(text):

    punct_pattern = str.maketrans('','',string.punctuation)

    return text.translate(punct_pattern)



train.text = train.text.apply(lambda x: remove_punctuations(x))

test.text = test.text.apply(lambda x: remove_punctuations(x))
def remove_stopwords(text):

    stop_words = set(stopwords.words('english'))

    if text is not None:

        tokens = [word for word in word_tokenize(text) if word not in stop_words]

        return ' '.join(tokens)

    return text



train.text = train.text.apply(lambda x: remove_stopwords(x))

test.text = test.text.apply(lambda x: remove_stopwords(x))
glove_emb = KeyedVectors.load_word2vec_format(

    '/kaggle/input/glove300d-word2vec-bin/glove.840B.300d_word2vec.bin', 

    binary=True)

glove_emb_dict = {}

for idx, word in tqdm(enumerate(glove_emb.vocab)):

    glove_emb_dict[word] = glove_emb.vectors[idx]
## The maximum number of words in a tweet will act as the maxlen for pad sequences

max_length = 25

# max(

#     max(train.text.apply(lambda x: len(x.split()))), 

#     max(test.text.apply(lambda x: len(x.split())))

# )

vocab_size = 10000

embedding_dim = 300

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"

validation_split = 0.10

num_epochs = 50
## Tokenize tweets

def tokenize_tweets(text):

    tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)

    tokenizer.fit_on_texts(text)

    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(text)

    training_padded = pad_sequences(training_sequences, maxlen = max_length, 

                                    padding = padding_type, truncating = trunc_type)

    return training_sequences, training_padded, word_index, tokenizer



## Create embedding layer

def create_embedding_layer(word_index, emb):

    num_words = min(vocab_size, len(word_index) + 1)

    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, index in word_index.items():

        if index >= vocab_size:

            continue

        embedding_vector = emb.get(word, np.zeros(embedding_dim, dtype='float32'))

        if embedding_vector is not None:

            embedding_matrix[index] = embedding_vector

    return tf.keras.layers.Embedding(num_words, embedding_dim, 

                           embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix), 

                           input_length = max_length, 

                           trainable = False)
training_sequences, training_padded, word_index, tokenizer = tokenize_tweets(train.text)

emb_layer = create_embedding_layer(word_index, glove_emb_dict)
training_padded.shape
## Create Model

model = tf.keras.Sequential()

model.add(emb_layer)

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation = 'relu'))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(64, activation = 'relu'))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(32, activation = 'relu'))

model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))



## set the optimization method, loss function and metrics to track

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),

              loss='binary_crossentropy',

              metrics=['accuracy'])



## Early Stopping Callback

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='val_auc',

    min_delta=0.001,

    verbose=1,

    patience=5,

    mode='max',

    restore_best_weights=True)
model.summary()
## Split into train and validation sets

X_train, X_valid, y_train, y_valid = train_test_split(training_padded, 

                                                      train.target.values, 

                                                      test_size = validation_split, 

                                                      random_state=123)
history = model.fit(X_train, y_train, batch_size = 100, epochs = num_epochs, 

                    callbacks = [early_stopping],

                    validation_data = (X_valid, y_valid))
def plot_loss(loss, val_loss, num_epochs):

    plt.figure()

    plt.plot(range(1, num_epochs + 1), loss, label="Training Loss")

    plt.plot(range(1, num_epochs + 1), val_loss, 'r--', label="Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Loss")

    plt.legend();



def plot_accuracy(acc, val_acc, num_epochs):

    plt.figure()

    plt.plot(range(1, num_epochs + 1), acc, label="Training Accuracy")

    plt.plot(range(1, num_epochs + 1), val_acc, 'r--', label="Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Accuracy")

    plt.legend();

    
## Plot training loss and accuracy

loss = history.history['loss']

val_loss = history.history['val_loss']

plot_loss(loss, val_loss, num_epochs)



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

plot_accuracy(acc, val_acc, num_epochs)
test_sequences = tokenizer.texts_to_sequences(test.text)

test_padded = pad_sequences(test_sequences, maxlen = max_length, 

                                    padding = padding_type, truncating = trunc_type)



predictions = model.predict(test_padded)

predictions = np.round(predictions).astype(int).flatten()
submission_df = pd.DataFrame({'id': test.id.values.tolist(), 'target': predictions})

submission_df
submission_df.target.value_counts()
# submission_df.to_csv('submission.csv',index=False)
train_tweets = IndexedList(train.text.apply(lambda x: x.split()).tolist())

train_tweets[10]
test_tweets = test.text.apply(lambda x: x.split()).tolist()

test_tweets[10]
## Generate embeddings for tweets

sif_model = uSIF(glove_emb, workers=2, lang_freq='en')

sif_model.train(train_tweets)
train_sif_embeddings = []

for idx in range(len(train_tweets)):

    train_sif_embeddings.append(sif_model.sv[idx])

len(train_sif_embeddings)
test_sif_embeddings = []

for tweet in test_tweets:

    test_sif_embeddings.append(sif_model.infer([(tweet, 0)]).flatten())

len(test_sif_embeddings)
train_sif = np.vstack(train_sif_embeddings)

test_sif = np.vstack(test_sif_embeddings)
## Split into train and validation sets

sif_X_train, sif_X_valid, sif_y_train, sif_y_valid = train_test_split(train_sif, 

                                                      train.target.values, 

                                                      test_size = validation_split, 

                                                      random_state=123)
## Create Model - SIF+GloVE

model2 = tf.keras.Sequential()

model2.add(tf.keras.layers.Dense(256, activation = 'relu'))

model2.add(tf.keras.layers.Dropout(0.5))

model2.add(tf.keras.layers.Dense(128, activation = 'relu'))

model2.add(tf.keras.layers.Dropout(0.5))

model2.add(tf.keras.layers.Dense(64, activation = 'relu'))

model2.add(tf.keras.layers.Dropout(0.5))

model2.add(tf.keras.layers.Dense(32, activation = 'relu'))

model2.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))



## set the optimization method, loss function and metrics to track

model2.compile(optimizer=tf.keras.optimizers.Adam(0.0001),

              loss='binary_crossentropy',

              metrics=['accuracy'])



## Early Stopping Callback

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='val_auc',

    min_delta=0.001,

    verbose=1,

    patience=5,

    mode='max',

    restore_best_weights=True)
sif_history = model2.fit(sif_X_train, sif_y_train, batch_size = 100, epochs = 50, 

                    callbacks = [early_stopping],

                    validation_data = (sif_X_valid, sif_y_valid))
## Plot training loss and accuracy

sif_loss = sif_history.history['loss']

sif_val_loss = sif_history.history['val_loss']

plot_loss(sif_loss, sif_val_loss, 50)



sif_acc = sif_history.history['accuracy']

sif_val_acc = sif_history.history['val_accuracy']

plot_accuracy(sif_acc, sif_val_acc, 50)
sif_predictions = model2.predict(test_sif)

sif_predictions = np.round(sif_predictions).astype(int).flatten()
sif_submission_df = pd.DataFrame({'id': test.id.values.tolist(), 'target': sif_predictions})

sif_submission_df
sif_submission_df.target.value_counts()
# sif_submission_df.to_csv('sif_submission.csv',index=False)
## Create Model

model3 = tf.keras.Sequential()

model3.add(emb_layer)

model3.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True)))

model3.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.4, recurrent_dropout=0.4)))

model3.add(tf.keras.layers.Dense(32, activation = 'relu'))

model3.add(tf.keras.layers.Dropout(0.5))

model3.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))



## set the optimization method, loss function and metrics to track

model3.compile(optimizer=tf.keras.optimizers.Adam(0.0001),

              loss='binary_crossentropy',

              metrics=['accuracy'])
model3.summary()
history3 = model3.fit(X_train, y_train, batch_size = 100, epochs = num_epochs, 

                    validation_data = (X_valid, y_valid))
## Plot training loss and accuracy

lstm_loss = history3.history['loss']

lstm_val_loss = history3.history['val_loss']

plot_loss(lstm_loss, lstm_val_loss, num_epochs)



lstm_acc = history3.history['accuracy']

lstm_val_acc = history3.history['val_accuracy']

plot_accuracy(lstm_acc, lstm_val_acc, num_epochs)
lstm_predictions = model3.predict(test_padded)

lstm_predictions = np.round(lstm_predictions).astype(int).flatten()
lstm_submission_df = pd.DataFrame({'id': test.id.values.tolist(), 'target': lstm_predictions})

lstm_submission_df
lstm_submission_df.target.value_counts()
lstm_submission_df.to_csv('lstm_submission.csv',index=False)
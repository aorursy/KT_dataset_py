import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers
train_df = pd.read_csv("../input/quora-question-pairs/train.csv")

test_df = pd.read_csv("../input/quora-question-pairs/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
## split to train and val

train_df, val_df = train_test_split(train_df, test_size=0.0001, random_state=2019)



## some config values 

embed_size = 500 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a question to use

epochs = 40 # num of epochs

dropout = 0.35 # dropout part



## fill up the missing values

train_X1 = train_df["question1"].fillna("_na_").values

train_X2 = train_df["question2"].fillna("_na_").values

val_X1 = val_df["question1"].fillna("_na_").values

val_X2 = val_df["question2"].fillna("_na_").values

test_X1 = test_df["question1"].fillna("_na_").values

test_X2 = test_df["question2"].fillna("_na_").values



## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(np.concatenate([train_X1, train_X2])))

train_X1 = tokenizer.texts_to_sequences(train_X1)

train_X2 = tokenizer.texts_to_sequences(train_X2)

val_X1 = tokenizer.texts_to_sequences(val_X1)

val_X2 = tokenizer.texts_to_sequences(val_X2)

test_X1 = tokenizer.texts_to_sequences(test_X1)

test_X2 = tokenizer.texts_to_sequences(test_X2)



## Pad the sentences 

train_X1 = pad_sequences(train_X1, maxlen=maxlen)

train_X2 = pad_sequences(train_X2, maxlen=maxlen)

val_X1 = pad_sequences(val_X1, maxlen=maxlen)

val_X2 = pad_sequences(val_X2, maxlen=maxlen)

test_X1 = pad_sequences(test_X1, maxlen=maxlen)

test_X2 = pad_sequences(test_X2, maxlen=maxlen)



## Get the target values

train_y = train_df['is_duplicate'].values

val_y = val_df['is_duplicate'].values
# Each instance will consist of two inputs: a single question1, and a single question2

question1_input = Input(shape=(maxlen,), name='question1')

question2_input = Input(shape=(maxlen,), name='question2')

question1_embedded = Embedding(max_features, embed_size, name='question1_embedded')(question1_input)

question2_embedded = Embedding(max_features, embed_size, name='question2_embedded')(question2_input)



# the first branch operates on the first input

x = Bidirectional(CuDNNGRU(75, return_sequences=True))(question1_embedded)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(dropout)(x)

x = Dense(1, activation="sigmoid")(x)

x = Model(inputs=question1_input, outputs=x)



# the second branch opreates on the second input

y = Bidirectional(CuDNNGRU(64, return_sequences=True))(question2_embedded)

y = GlobalMaxPool1D()(y)

y = Dense(16, activation="relu")(y)

y = Dropout(dropout)(y)

y = Dense(1, activation="sigmoid")(y)

y = Model(inputs=question2_input, outputs=y)



# combine the output of the two branches

combined = layers.concatenate([x.output, y.output])

 

# apply a FC layer and then a regression prediction on the

# combined outputs

z = Dense(16, activation="relu")(combined)

z = Dropout(dropout)(z)

z = Dense(1, activation="sigmoid")(z)



model = Model(

    inputs = [question1_input, question2_input],

    outputs = z,

)



model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])



model.summary()
## Train the model 

model.fit([train_X1, train_X2], train_y, batch_size=1024, epochs=4, validation_data=([val_X1, val_X2], val_y))
pred_noemb_val_y = model.predict([val_X1, val_X2], batch_size=1024, verbose=1)

print("log_loss score is {0}".format(metrics.log_loss(val_y, pred_noemb_val_y)))
pred_noemb_test_y = model.predict([test_X1, test_X2], batch_size=1024, verbose=1)
del model, question1_input, question2_input, x, y, z

import gc; gc.collect()

time.sleep(10)
EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector



# Each instance will consist of two inputs: a single question1, and a single question2

question1_input = Input(shape=(maxlen,), name='question1')

question2_input = Input(shape=(maxlen,), name='question2')

question1_embedded = Embedding(max_features, embed_size, weights=[embedding_matrix], name='question1_embedded')(question1_input)

question2_embedded = Embedding(max_features, embed_size, weights=[embedding_matrix], name='question2_embedded')(question2_input)



# the first branch operates on the first input

x = Bidirectional(CuDNNGRU(64, return_sequences=True))(question1_embedded)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(dropout)(x)

x = Dense(1, activation="sigmoid")(x)

x = Model(inputs=question1_input, outputs=x)



# the second branch opreates on the second input

y = Bidirectional(CuDNNGRU(64, return_sequences=True))(question2_embedded)

y = GlobalMaxPool1D()(y)

y = Dense(16, activation="relu")(y)

y = Dropout(dropout)(y)

y = Dense(1, activation="sigmoid")(y)

y = Model(inputs=question2_input, outputs=y)



# combine the output of the two branches

combined = layers.concatenate([x.output, y.output])

 

# apply a FC layer and then a regression prediction on the

# combined outputs

z = Dense(16, activation="relu")(combined)

z = Dropout(dropout)(z)

z = Dense(1, activation="sigmoid")(z)



model = Model(

    inputs = [question1_input, question2_input],

    outputs = z,

)



model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])



model.summary()
model.fit([train_X1, train_X2], train_y, batch_size=1024, epochs=epochs, validation_data=([val_X1, val_X2], val_y))
pred_glove_val_y = model.predict([val_X1, val_X2], batch_size=1024, verbose=1)

print("log_loss score is {0}".format(metrics.log_loss(val_y, pred_glove_val_y)))
pred_glove_test_y = model.predict([test_X1, test_X2], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, question1_input, question2_input, x, y, z

import gc; gc.collect()

time.sleep(10)
EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf8', errors='ignore') if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

# Each instance will consist of two inputs: a single question1, and a single question2

question1_input = Input(shape=(maxlen,), name='question1')

question2_input = Input(shape=(maxlen,), name='question2')

question1_embedded = Embedding(max_features, embed_size, weights=[embedding_matrix], name='question1_embedded')(question1_input)

question2_embedded = Embedding(max_features, embed_size, weights=[embedding_matrix], name='question2_embedded')(question2_input)



# the first branch operates on the first input

x = Bidirectional(CuDNNGRU(64, return_sequences=True))(question1_embedded)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(dropout)(x)

x = Dense(1, activation="sigmoid")(x)

x = Model(inputs=question1_input, outputs=x)



# the second branch opreates on the second input

y = Bidirectional(CuDNNGRU(64, return_sequences=True))(question2_embedded)

y = GlobalMaxPool1D()(y)

y = Dense(16, activation="relu")(y)

y = Dropout(dropout)(y)

y = Dense(1, activation="sigmoid")(y)

y = Model(inputs=question2_input, outputs=y)



# combine the output of the two branches

combined = layers.concatenate([x.output, y.output])

 

# apply a FC layer and then a regression prediction on the

# combined outputs

z = Dense(16, activation="relu")(combined)

z = Dropout(dropout)(z)

z = Dense(1, activation="sigmoid")(z)



model = Model(

    inputs = [question1_input, question2_input],

    outputs = z,

)



model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])



model.summary()
model.fit([train_X1, train_X2], train_y, batch_size=1024, epochs=epochs, validation_data=([val_X1, val_X2], val_y))
pred_fasttext_val_y = model.predict([val_X1, val_X2], batch_size=1024, verbose=1)

print("log_loss score is {0}".format(metrics.log_loss(val_y, pred_fasttext_val_y)))
pred_fasttext_test_y = model.predict([test_X1, test_X2], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, question1_input, question2_input, x, y, z

import gc; gc.collect()

time.sleep(10)
EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

# Each instance will consist of two inputs: a single question1, and a single question2

question1_input = Input(shape=(maxlen,), name='question1')

question2_input = Input(shape=(maxlen,), name='question2')

question1_embedded = Embedding(max_features, embed_size, weights=[embedding_matrix], name='question1_embedded')(question1_input)

question2_embedded = Embedding(max_features, embed_size, weights=[embedding_matrix], name='question2_embedded')(question2_input)



# the first branch operates on the first input

x = Bidirectional(CuDNNGRU(64, return_sequences=True))(question1_embedded)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(dropout)(x)

x = Dense(1, activation="sigmoid")(x)

x = Model(inputs=question1_input, outputs=x)



# the second branch opreates on the second input

y = Bidirectional(CuDNNGRU(64, return_sequences=True))(question2_embedded)

y = GlobalMaxPool1D()(y)

y = Dense(16, activation="relu")(y)

y = Dropout(dropout)(y)

y = Dense(1, activation="sigmoid")(y)

y = Model(inputs=question2_input, outputs=y)



# combine the output of the two branches

combined = layers.concatenate([x.output, y.output])

 

# apply a FC layer and then a regression prediction on the

# combined outputs

z = Dense(16, activation="relu")(combined)

z = Dropout(dropout)(z)

z = Dense(1, activation="sigmoid")(z)



model = Model(

    inputs = [question1_input, question2_input],

    outputs = z,

)



model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])



model.summary()
model.fit([train_X1, train_X2], train_y, batch_size=1024, epochs=epochs, validation_data=([val_X1, val_X2], val_y))
pred_paragram_val_y = model.predict([val_X1, val_X2], batch_size=1024, verbose=1)

print("log_loss score is {0}".format(metrics.log_loss(val_y, pred_paragram_val_y)))
pred_paragram_test_y = model.predict([test_X1, test_X2], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, question1_input, question2_input, x, y, z

import gc; gc.collect()

time.sleep(10)
pred_val_y = 0.33*pred_glove_val_y + 0.33*pred_fasttext_val_y + 0.34*pred_paragram_val_y

print("log_loss score is {0}".format(metrics.log_loss(val_y, pred_val_y)))
pred_test_y = 0.33*pred_glove_test_y + 0.33*pred_fasttext_test_y + 0.34*pred_paragram_test_y

out_df = pd.DataFrame({"test_id":test_df["test_id"].values})

out_df['is_duplicate'] = pred_test_y

out_df.to_csv("submission.csv", index=False)
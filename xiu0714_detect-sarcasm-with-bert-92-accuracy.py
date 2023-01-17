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
# import packages

import json

import os 

import tensorflow as tf

import sklearn

import seaborn as sbs

import sklearn.naive_bayes 

import sklearn.model_selection

import sklearn.metrics
json_1 = '../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json'

json_2 = '../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json'

def load_json(jfile):

    data = []

    with open(jfile) as f:

        for line in f.readlines():

            j = json.loads(line)

            url, headline, sarcastic = j['article_link'], j['headline'], j['is_sarcastic']

            data.append([url, headline, sarcastic])

    return pd.DataFrame(data, columns=['article_link', 'headline', 'is_sarcastic'])



print("✅✅✅ SESSION DONE")
df = load_json(json_1)

df2 = load_json(json_2)

df2
df.is_sarcastic.value_counts(normalize=True), df.is_sarcastic.value_counts()
# A birdview of headline length. Seems the majority has a length of 70, that's about 5 to 15 words, which makes sence.

sbs.distplot(df.headline.str.len())

vocab_size = 10000 # max_features 

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

tokenizer.fit_on_texts(df.headline)

train_inputs = tokenizer.texts_to_sequences(df.headline)

sbs.distplot([len(l) for l in train_inputs])
train_inputs = tf.keras.preprocessing.sequence.pad_sequences(train_inputs, padding='post', maxlen=20)

train_labels = df['is_sarcastic']



# Split data into train /validation 

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train_inputs, train_labels, test_size=0.2, random_state=0)

train_inputs[0]
nb = sklearn.naive_bayes.MultinomialNB()

nb.fit(X_train, y_train)
y_preds = nb.predict(X_val)

print(f"Accuracy score", sklearn.metrics.accuracy_score(y_val, y_preds))

print(f"Classification report\n", sklearn.metrics.classification_report(y_val, y_preds))
max_len = 20

text_input = tf.keras.Input(shape=(max_len, ))

embed_text = tf.keras.layers.Embedding(vocab_size, 128)(text_input)



net = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(embed_text)

net = tf.keras.layers.GlobalMaxPool1D()(net)

net = tf.keras.layers.Dense(64, activation='relu')(net)

net = tf.keras.layers.Dropout(0.4)(net)

net = tf.keras.layers.Dense(32, activation='relu')(net)

net = tf.keras.layers.Dropout(0.4)(net)



output = tf.keras.layers.Dense(1, activation='sigmoid')(net)

model = tf.keras.models.Model(text_input, output)

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



mc = tf.keras.callbacks.ModelCheckpoint('model_best.hdf5', monitor='val_accuracy', 

                                        verbose=1, save_best_only=True, mode='max')

es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5)

    

history = model.fit(X_train, y_train,

                    epochs=30, batch_size=256, callbacks=[mc, es], 

                    validation_split=0.1, verbose=1)
model = tf.keras.models.load_model('model_best.hdf5')

y_preds = model.predict(X_val, batch_size=1024).round().astype(int)

print("Test accracy score", sklearn.metrics.accuracy_score(y_val, y_preds))
%%time

import gensim.downloader as api



def embed_word_vector(word_index, pretrained='word2vec-google-news-300'):

    embed_size = 300 # Google news vector is 300-dimensional

    vector = api.load(pretrained)

    zeros = [0] * embed_size

    matrix = np.zeros((vocab_size, embed_size)) 

    

    for word, i in word_index.items():

        if i >= vocab_size or word not in vector: continue 

        matrix[i] = vector[word]

    

    print("Embed word vector completed.")

    return matrix
%%time

pretrained = 'glove-wiki-gigaword-300'

matrix = embed_word_vector(tokenizer.word_index, pretrained=pretrained)
max_len = 20

text_input = tf.keras.Input(shape=(max_len, ))

embed_text = tf.keras.layers.Embedding(vocab_size, 300, weights=[matrix], trainable=False)(text_input)



net = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(embed_text)

net = tf.keras.layers.GlobalMaxPool1D()(net)

net = tf.keras.layers.Dense(64, activation='relu')(net)

net = tf.keras.layers.Dropout(0.4)(net)

net = tf.keras.layers.Dense(32, activation='relu')(net)

net = tf.keras.layers.Dropout(0.4)(net)



output = tf.keras.layers.Dense(1, activation='sigmoid')(net)

model = tf.keras.models.Model(text_input, output)

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



mc = tf.keras.callbacks.ModelCheckpoint('model_best_embed.hdf5', monitor='val_accuracy', 

                                        verbose=1, save_best_only=True, mode='max')

es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5)

    

history = model.fit(X_train, y_train,

                    epochs=30, batch_size=256, callbacks=[mc, es], 

                    validation_split=0.1, verbose=1)
model = tf.keras.models.load_model('model_best_embed.hdf5')

y_preds = model.predict(X_val, batch_size=1024).round().astype(int)

print("Test accracy score", sklearn.metrics.accuracy_score(y_val, y_preds))
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
%%time

import tensorflow_hub as hub 

import tokenization



module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)



def build_model(bert_layer, max_len=512):

    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)

    net = tf.keras.layers.Dropout(0.2)(net)

    net = tf.keras.layers.Dense(32, activation='relu')(net)

    net = tf.keras.layers.Dropout(0.2)(net)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(net)

    

    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
max_len = 100

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(df.headline, df['is_sarcastic'], 

                                                                          test_size=0.1, random_state=0)

X_train = bert_encode(X_train, tokenizer, max_len=max_len)

X_val = bert_encode(X_val, tokenizer, max_len=max_len)
model = build_model(bert_layer, max_len=max_len)

model.summary()
%%time

checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)



train_history = model.fit(

    X_train, y_train, 

    validation_split=0.1,

    epochs=3,

    callbacks=[checkpoint, earlystopping],

    batch_size=16,

    verbose=1

)
%%time

model.load_weights('model.h5')

y_preds = model.predict(X_val).round().astype(int)

print("Validation accuracy: ", sklearn.metrics.accuracy_score(y_val, y_preds))
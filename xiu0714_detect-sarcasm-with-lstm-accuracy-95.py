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
json_1 = '../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json'
json_2 = '../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json'
import json
import os 
import tensorflow as tf
import sklearn
import seaborn as sbs
import sklearn.naive_bayes 
import sklearn.model_selection
import sklearn.metrics
import json

def load_json(jfile):
    data = []
    with open(jfile) as f:
        for line in f.readlines():
            j = json.loads(line)
            url, headline, sarcastic = j['article_link'], j['headline'], j['is_sarcastic']
            data.append([url, headline, sarcastic])
    return pd.DataFrame(data, columns=['article_link', 'headline', 'is_sarcastic'])

print("✅✅✅ SESSION DONE")
df = pd.concat([load_json(json_1), load_json(json_2)])
sbs.distplot(df.headline.str.len())
vocab_size = 30000  # There are 30884 words in total
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(df.headline)
print("Vocabulary size", len(tokenizer.word_index))

train_inputs = tokenizer.texts_to_sequences(df.headline)
sbs.distplot([len(l) for l in train_inputs])
train_inputs = tf.keras.preprocessing.sequence.pad_sequences(train_inputs, padding='post', maxlen=20)
train_labels = df['is_sarcastic']

# Split data into train /validation 
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train_inputs, train_labels, 
                                                                          test_size=0.2, random_state=0)
train_inputs[0]
max_len = 20
text_input = tf.keras.Input(shape=(max_len, ))
embed_text = tf.keras.layers.Embedding(vocab_size, 128)(text_input)

net = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(embed_text)
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
print("Test accuracy score", sklearn.metrics.accuracy_score(y_val, y_preds))
import tensorflow_hub as hub

hub_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
hub_layer = hub.KerasLayer(hub_url, output_shape=[128], input_shape=[],
                           dtype=tf.string, trainable=True)
# Split data into train /validation 
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(df.headline.values, df['is_sarcastic'], 
                                                                          test_size=0.2, random_state=0)
%%time 
# input = tf.keras.layers.Input(shape=(), name="Input", dtype=tf.string)
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1))

model.summary()

%%time
mc = tf.keras.callbacks.ModelCheckpoint('model_best_embed.hdf5', monitor='val_accuracy', 
                                        verbose=1, save_best_only=True, mode='max')
es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5)
    
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train,
                    epochs=30, batch_size=256, callbacks=[mc, es],
                    validation_split=0.1, verbose=1)

model.load_weights('model_best_embed.hdf5')
y_preds = model.predict(X_val, batch_size=1024)
y_preds = (y_preds >= 0.5).astype(int)
print("Test accracy score", sklearn.metrics.accuracy_score(y_val, y_preds))
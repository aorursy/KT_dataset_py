# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from transformers import AutoTokenizer, AutoModel

import tensorflow as tf

import transformers

import time





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
bert_checkpoint = "bert-base-multilingual-cased" #12-layer, 768-hidden, 12-heads, 110M parameters.

xlm_checkpoint = "xlm-mlm-100-1280"

xlm_roberta_checkpoint = "xlm-roberta-base" #~125M parameters with 12-layers, 768-hidden-state, 3072 feed-forward hidden-state, 8-heads

max_len = 128
bert_tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)

xlm_tokenizer = AutoTokenizer.from_pretrained(xlm_checkpoint)

xlm_roberta_tokenizer = AutoTokenizer.from_pretrained(xlm_roberta_checkpoint)
#создадим функцию токенизации датасета

def encode_comments(dataframe, tokenizer, max_len=max_len):

    pos = 0

    start = time.time()

    

    while pos < len(dataframe):

        temp = dataframe[pos:pos+10000].copy()

        res = tokenizer.batch_encode_plus(temp.comment_text.values,

                                          pad_to_max_length=True,

                                          max_length = max_len,

                                          return_attention_masks = False

                                         )

        if pos == 0:

            ids = np.array(res["input_ids"])

            labels = temp.toxic.values

        else:

            ids = np.concatenate((ids, np.array(res["input_ids"])))

            labels = np.concatenate((labels, temp.toxic.values))

        pos+=10000

        print("Processed", pos, "elements")

    return ids, labels
def make_model(transformer):

    

    inp = tf.keras.layers.Input(shape=(max_len,), dtype="int32")

    X = transformer(inp)[0]

    cls_token = X[:,0,:]

    X = tf.keras.layers.Dropout(0.3)(cls_token)

    X = tf.keras.layers.Dense(1, activation="sigmoid")(X)

    model = tf.keras.Model(inputs=inp, outputs=X)

    return model

    
train_path = "../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv"

val_path = "../input/jigsaw-multilingual-toxic-comment-classification/validation.csv"

train = pd.read_csv(train_path, usecols=["comment_text", "toxic"])

val = pd.read_csv(val_path, usecols=["comment_text", "toxic"])
ids_1, labels_1 = encode_comments(train, tokenizer=bert_tokenizer)

ids_2, labels_2 = encode_comments(train, tokenizer=xlm_tokenizer)

ids_3, labels_3 = encode_comments(train, tokenizer=xlm_roberta_tokenizer)



val_ids_1, val_labels_1 = encode_comments(val, tokenizer=bert_tokenizer)

val_ids_2, val_labels_2 = encode_comments(val, tokenizer=xlm_tokenizer)

val_ids_3, val_labels_3 = encode_comments(val, tokenizer=xlm_roberta_tokenizer)

len(val)
encode_comments(val, tokenizer=bert_tokenizer)
AUTO = tf.data.experimental.AUTOTUNE



tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

print(tpu_strategy.num_replicas_in_sync)
batch_size = 16 * tpu_strategy.num_replicas_in_sync

with tpu_strategy.scope():

    bert = transformers.TFBertModel.from_pretrained(bert_checkpoint)

    m1 = make_model(bert)

    m1.compile(optimizer=tf.keras.optimizers.Adam(3e-5), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])

    m1.summary()
h1 = m1.fit(ids_1, labels_1, batch_size=batch_size, validation_data=(val_ids_1, val_labels_1), epochs=1)
del m1
with tpu_strategy.scope():

    xlm_r = transformers.TFXLMRobertaModel.from_pretrained("jplu/tf-xlm-roberta-base")

    m3 = make_model(xlm_r)

    m3.compile(optimizer=tf.keras.optimizers.Adam(3e-5), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])

    m3.summary()
with tpu_strategy.scope():

    xlm = transformers.TFXLMModel.from_pretrained(xlm_checkpoint)

    m2 = make_model(xlm)

    m2.compile(optimizer=tf.keras.optimizers.Adam(3e-5), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])

    m2.summary()
h3 = m3.fit(ids_3, labels_3, batch_size=batch_size, validation_data=(val_ids_3, val_labels_3), epochs=1)
del m3
h2 = m2.fit(ids_2, labels_2, batch_size=batch_size, validation_data=(val_ids_2, val_labels_2), epochs=1)
del m2
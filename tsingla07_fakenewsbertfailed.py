# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# from transformers import *

# import tokenizers

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

import tokenization
def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for i, text in enumerate(texts):

        

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

        if i%100 == 0:

            print(i,"/", len(texts))

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def build_model(bert_layer, max_len=512):

#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

#     tf.config.experimental_connect_to_cluster(tpu)

#     tf.tpu.experimental.initialize_tpu_system(tpu)



#     # instantiate a distribution strategy

#     tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)



#     # instantiating the model in the strategy scope creates the model on the TPU

#     with tpu_strategy.scope():



    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)



    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])



    return model
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train =pd.read_csv('../input/fake-news/train.csv')

train.info()

train.fillna('null', inplace = True)
test = pd.read_csv('../input/fake-news/test.csv')

test.info()

test.fillna('null', inplace=True)
train_input = bert_encode(train.text.values, tokenizer, max_len=160)

test_input = bert_encode(test.text.values, tokenizer, max_len=160)

train_labels = train.label.values
np.save('train_input.npy', train_input)

np.save('test_input.npy', test_input)
# a = bert_encode(train.text.values[:1], tokenizer, max_len=160)

# a = (tf.convert_to_tensor(a[0], tf.int32), tf.convert_to_tensor(a[1], tf.int32), tf.convert_to_tensor(a[2], tf.int32))
model = build_model(bert_layer, max_len=160)

model.summary()
train_history = model.fit(

    [train_input[0], train_input[1], train_input[2]], train_labels,

    validation_split=0.2,

    epochs=2,

    batch_size=8,

    verbose = 2

)



model.save('model.h5')
a = test

a['label'] = model.predict([test_input[0], test_input[1], test_input[2]])

a['label'] = a['label'].apply(round)

a.head()

a[['id', 'label']].to_csv('submission1.csv',index=False)
# # a = bert_encode(train.text.values[:2], tokenizer, max_len=160)

# max_len = 10

# input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

# input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

# segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



# _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

# clf_output = sequence_output[:, 0, :]

# out = Dense(1, activation='sigmoid')(clf_output)

# model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)



# model(a)
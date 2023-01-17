# We will use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tensorflow as tf

import numpy as np

import pandas as pd

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Input, Bidirectional, SpatialDropout1D, Embedding, add, concatenate

from tensorflow.keras.layers import GRU, GlobalAveragePooling1D, LSTM, GlobalMaxPooling1D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import tensorflow_hub as hub



import tokenization
train = pd.read_csv('../input/preprocesseddata/train.csv')

test = pd.read_csv('../input/preprocesseddata/test.csv')

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
def bert_encoder(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

        

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segments_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segments_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def build_model(bert_layer, max_len=512):

 

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    x = SpatialDropout1D(0.3)(sequence_output)

    x = Bidirectional(GRU(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(GRU(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([GlobalMaxPooling1D()(x),GlobalAveragePooling1D()(x),])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=result)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encoder(train.text.values, tokenizer, max_len=160)

test_input = bert_encoder(test.text.values, tokenizer, max_len=160)

train_labels = train.target.values
import gc

NUM_MODELS = 1

    

BATCH_SIZE = 16

LSTM_UNITS = 64

EPOCHS = 10

DENSE_HIDDEN_UNITS = 256

checkpoint_predictions = []

checkpoint_val_pred = []

weights = []



for model_idx in range(NUM_MODELS):

    model = build_model(bert_layer, max_len=160)

    for global_epoch in range(EPOCHS):

        model.fit(

            train_input, train_labels,

            batch_size=BATCH_SIZE,

            validation_split=.25,

            epochs=1,

            verbose=1,

            callbacks=[

                LearningRateScheduler(lambda epoch: 2e-6 * (0.6** global_epoch))

            ]

        )

        checkpoint_predictions.append(model.predict(test_input, batch_size=64).flatten())

        weights.append(2 ** global_epoch)

    del model

    gc.collect()
test_pred = np.average(checkpoint_predictions, weights=weights, axis=0)
submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)
import os

import numpy as np
from sklearn import metrics
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
#Define encoder.
def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])
#'''
def build_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_lang_tags = Input(shape=(4,), dtype=tf.float32, name="input_lang_tags")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    x = Concatenate()([cls_token, input_lang_tags])
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input_word_ids, input_lang_tags], outputs=out)
    
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    
    return model

'''
#Build a pure text model where language information is not considered.
def build_model_PT(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    
    return model
#'''
AUTO = tf.data.experimental.AUTOTUNE
# Configuration
EPOCHS = 2
BATCH_SIZE = 16 * strategy.num_replicas_in_sync#16
MAX_LEN = 192
# First load the real tokenizer
MODEL = 'jplu/tf-xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
def lang_embed(lang, tran):
    lang_codes = {'en':'000', 'es':'100', 'fr':'010',
                  'it':'001', 'pt':'110', 'ru':'101',
                  'tr':'011'}
    tran_codes = {'orig':'0', 'tran':'1'}
    vec = lang_codes[lang]+tran_codes[tran]
    vec = [int(v) for v in vec]
    return vec
def text_process(text):
    ws = text.split(' ')
    if(len(ws)>160):
        text = ' '.join(ws[:160]) + ' ' + ' '.join(ws[-32:])
    return text

#Build the original validation corpus.
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
valid['comment_text'] = valid['comment_text'].apply(lambda x: text_process(x))

x_valid = regular_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)
lang_tag_valid = np.array([lang_embed(row['lang'], 'orig') for _, row in valid.iterrows()])
y_valid = valid.toxic.values

gap = 128#valid.shape[0]%BATCH_SIZE
x_valid = np.concatenate((x_valid, x_valid[-gap:]))
lang_tag_valid = np.concatenate((lang_tag_valid, lang_tag_valid[-gap:]))
y_valid = np.concatenate((y_valid, y_valid[-gap:]))
print(y_valid.shape)
'''
#Build the translated validation corpus.
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_valid_translated.csv')
valid['comment_text'] = valid['comment_text'].apply(lambda x: text_process(x))

x_tran_valid = regular_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)
lang_tag_tran_valid = np.array([lang_embed('en', 'tran') for _, row in valid.iterrows()])
y_tran_valid = valid.toxic.values

gap = 128#valid.shape[0]%BATCH_SIZE
x_tran_valid = np.concatenate((x_tran_valid, x_tran_valid[-gap:]))
lang_tag_tran_valid = np.concatenate((lang_tag_tran_valid, lang_tag_tran_valid[-gap:]))
y_tran_valid = np.concatenate((y_tran_valid, y_tran_valid[-gap:]))
print(y_tran_valid.shape)
'''
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
#Build the original and translated test corpora.
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
test['content'] = test['content'].apply(lambda x: text_process(x))

#tran_test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')
#tran_test['translated'] = tran_test['translated'].apply(lambda x: text_process(x))
                       
x_test = regular_encode(test.content.values, tokenizer, maxlen=MAX_LEN)
lang_tag_test = np.array([lang_embed(row['lang'], 'orig') for _, row in test.iterrows()])

#x_tran_test = regular_encode(tran_test.translated.values, tokenizer, maxlen=MAX_LEN)
#lang_tag_tran_test = np.array([lang_embed('en', 'tran') for _, row in tran_test.iterrows()])
mybest = pd.read_csv('/kaggle/input/mybest/sub9521.csv')
mybest['orig'] = test['content']
mybest['lang'] = test['lang']
mybest['tran'] = ''#tran_test['translated']

out = []
for _, row in mybest.iterrows():
    #if row['lang'] not in ('fr', 'ru', 'pt'):#Only gather those not in validation?
        #continue
    if(row['toxic']>=0.5):
        out.append([row['orig'], row['tran'], row['lang'], 1])
    elif(row['toxic']<0.5):
        out.append([row['orig'], row['tran'], row['lang'], 0])

train = pd.DataFrame(out, columns=['orig', 'tran', 'lang', 'toxic'])
train['orig'] = train['orig'].apply(lambda x: text_process(x))
#train['tran'] = train['tran'].apply(lambda x: text_process(x))
print(train.shape)
%%time 
x_orig_train = regular_encode(train.orig.values, tokenizer, maxlen=MAX_LEN)
lang_tag_orig_train = np.array([lang_embed(row['lang'], 'orig') for _, row in train.iterrows()])

#x_tran_train = regular_encode(train.tran.values, tokenizer, maxlen=MAX_LEN)
#lang_tag_tran_train = np.array([lang_embed('en', 'tran') for _, row in train.iterrows()])

y_train = train.toxic.values

gap = 128#train.shape[0]%BATCH_SIZE
x_orig_train = np.concatenate((x_orig_train, x_orig_train[-gap:]))
lang_tag_orig_train = np.concatenate((lang_tag_orig_train, lang_tag_orig_train[-gap:]))

#x_tran_train = np.concatenate((x_tran_train, x_tran_train[-gap:]))
#lang_tag_tran_train = np.concatenate((lang_tag_tran_train, lang_tag_tran_train[-gap:]))

y_train = np.concatenate((y_train, y_train[-gap:]))
print(y_train.shape)
%%time
with strategy.scope():
    transformer_layer = TFAutoModel.from_pretrained(MODEL)
    #model = build_model_PT(transformer_layer, max_len=MAX_LEN)
    model = build_model(transformer_layer, max_len=MAX_LEN)
#model.summary()
#model.load_weights('/kaggle/input/basemodels/mg2m.h5')
model.load_weights('/kaggle/input/basemodels/mixmoriggen1.h5')
#model.load_weights('/kaggle/input/en2m1211/en2m1211.h5')
#model.load_weights('/kaggle/input/mixmodel0/mixm0.h5')
#'''#First train on best orignal data.
history = model.fit([x_orig_train[:63872],lang_tag_orig_train[:63872]], y_train[:63872],
                    validation_data=([x_valid[:8064],lang_tag_valid[:8064]], y_valid[:8064]),
                    batch_size=BATCH_SIZE, epochs=1, verbose=1)
'''
#First train on best orignal data.
history = model.fit(x_orig_train[:63872], y_train[:63872],
                    validation_data=(x_valid[:8064], y_valid[:8064]),
                    batch_size=BATCH_SIZE, epochs=1, verbose=1)
#'''
#'''#Fine train on validation data.#
history = model.fit([x_valid[:8064],lang_tag_valid[:8064]], y_valid[:8064],
                    validation_data=([x_valid[:8064],lang_tag_valid[:8064]], y_valid[:8064]),
                    batch_size=BATCH_SIZE, epochs=1, verbose=1)
'''
history = model.fit(x_valid[:8064], y_valid[:8064],
                    validation_data=(x_valid[:8064], y_valid[:8064]),
                    batch_size=BATCH_SIZE, epochs=2, verbose=1)
#'''
model.save_weights("/kaggle/working/mixgn2mp4.h5")
from sklearn.metrics import roc_auc_score
sub['toxic'] = model.predict([x_test, lang_tag_test], verbose=1)
#sub['toxic'] = model.predict(x_test, verbose=1)
sub.to_csv('submission.csv', index=False)
score1 = roc_auc_score(mybest.toxic.round().astype(int), sub.toxic.values)
score2 = roc_auc_score(sub.toxic.round().astype(int), mybest.toxic.values)
print('p: %2.4f %2.4f'%(100*score1, 100*score2))
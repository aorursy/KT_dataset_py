import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from transformers import TFAutoModel, AutoTokenizer,BertTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from sklearn.model_selection import KFold

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)

def build_model(transformer, max_len=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(6, activation='softmax')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
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
AUTO = tf.data.experimental.AUTOTUNE

# Data access
# GCS_DS_PATH = KaggleDatasets().get_gcs_path()

# Configuration
EPOCHS = 2
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192

# tokenizer = AutoTokenizer.from_pretrained("voidful/albert_chinese_xlarge")
# MODEL = 'jplu/tf-xlm-roberta-large'
MODEL = "hfl/chinese-roberta-wwm-ext"
MODEL = 'hfl/chinese-roberta-wwm-ext-large'
MODEL = '../input/chinese-roberta-wwm-ext-l12-h768-a12'
# MODEL = '../input/hflchineserobertawwmext'
MODEL = 'bert-base-chinese'
# First load the real tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
# read weibo data
usual_train = pd.read_excel('../input/weibo-sentiment-analysis/usual_train.xlsx')
virus_train = pd.read_excel('../input/weibo-sentiment-analysis/virus_train.xlsx')
usual_test = pd.read_excel('../input/weibo-sentiment-analysis/usual_eval.xlsx')
virus_test = pd.read_excel('../input/weibo-sentiment-analysis/virus_eval.xlsx')

usual_train['type'] = 'usual'
usual_test['type'] = 'usual'
virus_train['type'] = 'virus'
virus_test['type'] = 'virus'
print(usual_train.head())
print(virus_train.head())
data = usual_train.append(virus_train)
data['文本'] = data['文本'].astype('str')
# 打乱数据
data = data.sample(frac=1,random_state=2020).reset_index(drop=True)
print(data.shape)
label_dict = {}
for label in data['情绪标签'].unique():
    label_dict[label] = len(label_dict)
print(label_dict)
data['情绪标签'] = data['情绪标签'].map(label_dict)
data['文本'] = data.apply(lambda x:'[疫情]'+x['文本'] if x.type=='virus' else '[普通]'+x['文本'],axis=1)
data
def regular_encode(texts, tokenizer, data_type, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    res = np.array(enc_di['input_ids'])
#     res2 = [np.insert(res[index],1,1) if data_type[index]=='usual' else np.insert(res[index],1,2) for index in range(len(texts))]
    return np.array(res)
kfold = KFold(n_splits=4, shuffle=True, random_state=2019)
for train_index, test_index in kfold.split(np.zeros(len(data))):
    train = data.loc[train_index,:].reset_index()
    val = data.loc[test_index,:].reset_index()
    x_train = regular_encode(train['文本'].values, tokenizer, train['type'].values,maxlen=MAX_LEN)
    x_valid = regular_encode(val['文本'].values, tokenizer, val['type'].values, maxlen=MAX_LEN)
    y_train = train['情绪标签'].values
    y_valid = val['情绪标签'].values
    train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, tf.keras.utils.to_categorical(y_train)))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
    valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, tf.keras.utils.to_categorical(y_valid)))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)
#     %%time
    with strategy.scope():
        transformer_layer = TFAutoModel.from_pretrained(MODEL)
        model = build_model(transformer_layer, max_len=MAX_LEN)
#     model.summary()
    n_steps = x_train.shape[0] // BATCH_SIZE
    train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        validation_data=valid_dataset,
        epochs=EPOCHS+1
    )
    val['pred'] = model.predict(valid_dataset, verbose=1).argmax(axis=1)
    val['match'] = val['pred']==val['情绪标签']
    print(val['match'].mean())
    print(val[val.type=='usual']['match'].mean())
    print(val[val.type=='virus']['match'].mean())
1
0.7713877281724214
0.7661302015369001
0.7878925807919891
2
0.7790851110622389
0.7720525264059378
0.8026819923371648
3
0.7819201583635764
0.7797515886770653
0.7888427846934071
4
0.7890685142417244
0.787350525860827
0.7946096654275093

x_test_usual = regular_encode(usual_test['文本'].values, tokenizer,usual_test['type'].values, maxlen=MAX_LEN)
x_test_virus = regular_encode(virus_test['文本'].values, tokenizer,virus_test['type'].values, maxlen=MAX_LEN)
# %%time 

# # x_train = regular_encode(train.comment_text.values, tokenizer, maxlen=MAX_LEN)
# # x_valid = regular_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)
# x_test = regular_encode(test.content.values, tokenizer, maxlen=MAX_LEN)

# y_train = train.toxic.values
# y_valid = valid.toxic.values
usual_test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test_usual)
    .batch(BATCH_SIZE)
)
virus_test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test_virus)
    .batch(BATCH_SIZE)
)
n_steps = x_valid.shape[0] // BATCH_SIZE
train_history_2 = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=n_steps,
    epochs=EPOCHS
)
usual_pred = model.predict(usual_test_dataset, verbose=1).argmax(axis=1)
virus_pred = model.predict(virus_test_dataset, verbose=1).argmax(axis=1)
id_label = {}
for key,value in label_dict.items():
    id_label[value] = key
usual_result = []
for index, id in enumerate(usual_test['数据编号']):
    line = {}
    line['id'] = id
    line['label'] = id_label[usual_pred[index]]
    usual_result.append(line)
virus_result = []
for index, id in enumerate(virus_test['数据编号']):
    line = {}
    line['id'] = id
    line['label'] = id_label[virus_pred[index]]
    virus_result.append(line)
with open('usual_result.txt', 'w', encoding='utf-8') as f:
    json.dump(usual_result, f)
with open('virus_result.txt', 'w', encoding='utf-8') as f:
    json.dump(virus_result, f)

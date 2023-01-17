from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets

import transformers



from tokenizers import BertWordPieceTokenizer

from tqdm import tqdm

import numpy as np



!pip install wandb



import os, time

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

from kaggle_datasets import KaggleDatasets



# We'll use a tokenizer for the BERT model from the modelling demo notebook.

!pip install bert-tensorflow

import bert.tokenization



print(tf.version.VERSION)
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
SEQUENCE_LENGTH = 128



DATA_PATH =  KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')

BERT_PATH = KaggleDatasets().get_gcs_path('bert-multi')

BERT_PATH_SAVEDMODEL = BERT_PATH + "/bert_multi_from_tfhub"



OUTPUT_PATH = "/kaggle/working"
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

sub2 = pd.read_csv('../input/ensemble/submission.csv')
def get_tokenizer(bert_path=BERT_PATH_SAVEDMODEL):

    """Get the tokenizer for a BERT layer."""

    bert_layer = tf.saved_model.load(bert_path)

    bert_layer = hub.KerasLayer(bert_layer, trainable=False)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

    cased = bert_layer.resolved_object.do_lower_case.numpy()

    tf.gfile = tf.io.gfile  # for bert.tokenization.load_vocab in tokenizer

    tokenizer = bert.tokenization.FullTokenizer(vocab_file, cased)

  

    return tokenizer



tokenizer = get_tokenizer()
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):

    """

    Encoder for encoding the text into sequence of integers for BERT Input

    """

    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)
#IMP DATA FOR CONFIG



AUTO = tf.data.experimental.AUTOTUNE





# Configuration

EPOCHS = 5

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 192
# First load the real tokenizer

tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# Save the loaded tokenizer locally

tokenizer.save_pretrained('.')

# Reload it with the huggingface tokenizers library

fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)

fast_tokenizer
x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_valid = fast_encode(valid.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)



y_train = train1.toxic.values

y_valid = valid.toxic.values
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
def build_model(transformer, max_len=512):

    """

    function for training the BERT model

    """

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    

    cls_token = sequence_output[:, 0, :]

    out = tf.keras.layers.Dense(192, activation='relu')(cls_token)

    out = tf.keras.layers.Dense(64, activation='relu')(out)

    out = tf.keras.layers.Dense(64, activation='relu')(out)

    out = Dense(1, activation='sigmoid')(out)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
%%time

with strategy.scope():

    transformer_layer = (

        transformers.TFDistilBertModel

        .from_pretrained('distilbert-base-multilingual-cased')

    )

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
n_steps = x_train.shape[0] // BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS

)
n_steps = x_valid.shape[0] // BATCH_SIZE

train_history_2 = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    epochs=EPOCHS*2

)
sub['toxic'] = model.predict(test_dataset, verbose=1)

#sub.to_csv('submission.csv', index=False)



sub1 = sub[['id', 'toxic']]
sub1.rename(columns={'toxic':'toxic1'}, inplace=True)

sub2.rename(columns={'toxic':'toxic2'}, inplace=True)

sub3 = pd.merge(sub1, sub2, how='left', on='id')



sub3['toxic'] = (sub3['toxic1'] * 0.1) + (sub3['toxic2'] * 0.9) #blend 1

sub3['toxic'] = (sub3['toxic2'] * 0.39) + (sub3['toxic'] * 0.61) #blend 2



sub3[['id', 'toxic']].to_csv('submission.csv', index=False)
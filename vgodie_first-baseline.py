# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import transformers

from transformers import TFBertModel

from tensorflow.keras import Model

from tensorflow.keras.layers import Input, Dense, Dropout

from tensorflow.keras.optimizers import Adam

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from transformers import BertTokenizer



tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")



sentence = "Hello what's up"

tokenized = tokenizer.encode(sentence)

print(tokenized)

print(tokenizer.convert_ids_to_tokens(tokenized))
DATA_PATH = "../input/jigsaw-multilingual-toxic-comment-classification/"

small_ds_processed_path = "jigsaw-toxic-comment-train-processed-seqlen128.csv"

val_path = "validation-processed-seqlen128.csv"
AUTO = tf.data.experimental.AUTOTUNE



tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

print(tpu_strategy.num_replicas_in_sync)
SEQUENCE_LENGTH = 128

BATCH_SIZE = 16 * tpu_strategy.num_replicas_in_sync
train = pd.read_csv(os.path.join(DATA_PATH, small_ds_processed_path))

val = pd.read_csv(os.path.join(DATA_PATH, val_path))

test = pd.read_csv(os.path.join(DATA_PATH, "test-processed-seqlen128.csv"))
train.head()
val.head()
train = train[["input_word_ids", "toxic"]]

val = val[["input_word_ids", "toxic"]]
print("train")

print(train.dtypes)

print("validation")

print(val.dtypes)
train_comments = train["input_word_ids"]

val_comments = val["input_word_ids"]

test_comments = test["input_word_ids"]
train_comments = train_comments.str.strip("()").str.split(",",expand=True).astype(int).values

val_comments = val_comments.str.strip("()").str.split(",",expand=True).astype(int).values

test_comments = test_comments.str.strip("()").str.split(",",expand=True).astype(int).values
train_labels = train["toxic"]

val_labels = val["toxic"]
BUFFER_SIZE = len(train_comments)

train_ds = (tf.data.Dataset.from_tensor_slices((train_comments, train_labels))

            .shuffle(BUFFER_SIZE)

            .repeat()

            .batch(BATCH_SIZE)

            .prefetch(AUTO)

           )



val_ds = (tf.data.Dataset.from_tensor_slices((val_comments, val_labels))

          .shuffle(BUFFER_SIZE)

          .batch(BATCH_SIZE)

          .prefetch(AUTO)

         )

from transformers import TFBertModel
def make_model(transformer):

    

    

    input_ids = Input(shape=(SEQUENCE_LENGTH,), name='input_token', dtype='int32')



    embed_layer = transformer(input_ids)[0]

    cls_token = embed_layer[:,0,:]

    X = Dropout(0.3)(cls_token)

    X = Dense(1, activation="sigmoid")(X)

    model = tf.keras.Model(inputs=input_ids, outputs = X)

    return model
with tpu_strategy.scope():

    bert = TFBertModel.from_pretrained("bert-base-multilingual-cased")

    model = make_model(bert)

    model.compile(optimizer=Adam(3e-5), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])

    model.summary()
N_STEPS = train_comments.shape[0] // BATCH_SIZE

VAL_STEPS = val_comments.shape[0] // BATCH_SIZE
EPOCHS = 2
history = model.fit(train_ds,

                    validation_data= val_ds,

                    epochs=EPOCHS,

                    steps_per_epoch=N_STEPS

                   )
history_plus = model.fit(val_ds,

                         epochs=EPOCHS,

                         steps_per_epoch=VAL_STEPS

                        )
sub = pd.read_csv(os.path.join('../input/jigsaw-multilingual-toxic-comment-classification/','sample_submission.csv'))

sub['toxic'] = model.predict(test_comments, verbose=1)

sub.to_csv('submission.csv', index=False)
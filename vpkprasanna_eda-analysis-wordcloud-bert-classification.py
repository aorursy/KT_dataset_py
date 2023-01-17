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
from matplotlib import rcParams

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from wordcloud import WordCloud
train = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
train.head()
df=train.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)

df=df.rename(columns={'v1':'labels','v2': 'text'})

df.head()
rcParams["figure.figsize"] = 10,15

df["labels"].value_counts().plot(kind="pie")
rcParams["figure.figsize"] = 15,8

df["labels"].value_counts().plot(kind="bar")
lb = LabelEncoder()

df["labels"] = lb.fit_transform(df["labels"])
df.head()
df["text_len"] = df["text"].apply(len)
df.head()
rcParams["figure.figsize"] = 10,6

sns.distplot(a=df['text_len'],kde=False)

plt.legend()
def generate_word_cloud(text):

    wordcloud = WordCloud(

        width = 3000,

        height = 2000,

        background_color = 'black').generate(str(text))

    fig = plt.figure(

        figsize = (40, 30),

        facecolor = 'k',

        edgecolor = 'k')

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()
real_data_text = df.text

generate_word_cloud(real_data_text)
!wget https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub

import tokenization
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
def bert_encode(texts, tokenizer, max_len=512):

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

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def build_model(bert_layer, max_len):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(200, activation='sigmoid')(clf_output)

    out1 = Dense(1,activation='sigmoid')(out)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out1)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(df.text.values, tokenizer, max_len=160)

# test_input = bert_encode(test.text.values, tokenizer, max_len=160)

train_labels = df.labels.values
model = build_model(bert_layer, max_len=512)

model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)



train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=5,

    callbacks=[checkpoint],

    batch_size=16

)



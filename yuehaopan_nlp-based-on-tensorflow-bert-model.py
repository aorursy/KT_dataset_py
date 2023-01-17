!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import numpy as np

import pandas as pd



import tensorflow as tf

import tensorflow_hub as hub

import tokenization

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint



import matplotlib

import matplotlib.pyplot as plt

#from wordcloud import WordCloud



import re

import string

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## Load csv files 

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
print('Missing location:', train[train.location != train.location].shape[0])

print('Missing keyword:', train[train.keyword != train.keyword].shape[0])
disasterCount = train[train.target == 1].shape[0]

nonDisasCount = train[train.target == 0].shape[0]



plt.rcParams['figure.figsize'] = (5,3)

plt.bar(2, disasterCount, width=1, label='Disaster', color='blue')

plt.bar(4, nonDisasCount, width=1, label='Non Disaster', color='red')

plt.legend()

plt.ylabel('# cases')

plt.title('Disaster / Non Disaster Count')

plt.show()
## To Do

# 1.World Cloud
# Clean URL: https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert/data

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)

def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



train['text']=train['text'].apply(lambda x : remove_URL(x))

train['text']=train['text'].apply(lambda x : remove_html(x))

train['text']=train['text'].apply(lambda x : remove_punct(x))



test['text']=test['text'].apply(lambda x : remove_URL(x))

test['text']=test['text'].apply(lambda x : remove_html(x))

test['text']=test['text'].apply(lambda x : remove_punct(x))
# Is this really useful?

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
## Targeted Cor

# https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data/comments

ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]

train[train['id'].isin(ids_with_target_error)]

train.at[train['id'].isin(ids_with_target_error),'target'] = 0

train[train['id'].isin(ids_with_target_error)]
def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

        # Token embedding [CLS],[SEP]    

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



def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"

bert_layer = hub.KerasLayer(module_url, trainable=True)

## Load BERT from the Tensorflow Hub
## Load tokenizer from the bert layer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
## Encode the text into tokens, masks, and segment flags

train_input = bert_encode(train.text.values, tokenizer, max_len=160)

test_input = bert_encode(test.text.values, tokenizer, max_len=160)

train_labels = train.target.values
model = build_model(bert_layer, max_len=160)

model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)



train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=5,

    callbacks=[checkpoint],

    batch_size=16

)
model.load_weights('model.h5')

test_pred = model.predict(test_input)
# Save output

from datetime import datetime

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')



submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission_BERT_' + timestamp + '.csv', index=False)

print("Your submission was successfully saved on " + timestamp)
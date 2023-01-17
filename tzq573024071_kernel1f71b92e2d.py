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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import re
import string
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
df_train= pd.read_csv('../input/nlp-getting-started/train.csv')
df_test=pd.read_csv('../input/nlp-getting-started/test.csv')
submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
df_train.head()
df_train.info()
# Check the distribution of the classes.
x = df_train.target.value_counts()
plt.bar(x.index, x)
plt.xlabel('Real vs. Fake')
plt.ylabel('number of tweets')
plt.title('distribution of classes')
plt.plot()
#check length of tweets from both classes
def length(text):
    return len(text)
df_train['length'] = df_train['text'].apply(length)
df_train['length'].head()
#visualize the length
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
df_train[df_train.target==1]['length'].plot.hist(ax=ax1).set_title('fake')
df_train[df_train.target==0]['length'].plot.hist(ax=ax2).set_title('real')
ax1.grid()
ax2.grid()
plt.plot()
#check stop words for fake class
corpus = []
for i in df_train[df_train['target']==0]['text'].str.split():
    for n in i:
        corpus.append(n)
dic = defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word] += 1
        
top_words = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]
x, y = zip(*top_words)
plt.bar(x, y)
#check stop words for real class
corpus = []
for i in df_train[df_train['target']==1]['text'].str.split():
    for n in i:
        corpus.append(n)
dic = defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word] += 1
        
top_words = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]
x, y = zip(*top_words)
plt.bar(x, y)
#NGram analysis
# trigram analysis
def get_top_tweet_trigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3,3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
plt.figure(figsize=(10,10))
top_tweet_trigrams = get_top_tweet_trigrams(df_train['text'], n = 10)
x, y = map(list, zip(*top_tweet_trigrams))
sns.barplot(x=y, y=x)
# remove URL
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
#remove HTML tages
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
#remove emoji
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
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
# remove punctuations
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
df_train['text'] = df_train['text'].apply(lambda x: remove_URL(x))
df_train['text'] = df_train['text'].apply(lambda x: remove_html(x))
df_train['text'] = df_train['text'].apply(lambda x: remove_emoji(x))
df_train['text'] = df_train['text'].apply(lambda x: remove_punct(x))
df_train.head()
df_test.head()
df_test['text'] = df_test['text'].apply(lambda x: remove_URL(x))
df_test['text'] = df_test['text'].apply(lambda x: remove_html(x))
df_test['text'] = df_test['text'].apply(lambda x: remove_emoji(x))
df_test['text'] = df_test['text'].apply(lambda x: remove_punct(x))
df_test.head()

# # correct spelling
# from spellchecker import SpellChecker

# spell = SpellChecker()
# def correct_spellings(text):
#     corrected_text = []
#     misspelled_words = spell.unknown(text.split())
#     for word in text.split():
#         if word in misspelled_words:
#             corrected_text.append(spell.correction(word))
#         else:
#             corrected_text.append(word)
#     return " ".join(corrected_text)
plt.figure(figsize=(10,10))
top_tweet_trigrams = get_top_tweet_trigrams(df_train['text'], n = 10)
x, y = map(list, zip(*top_tweet_trigrams))
sns.barplot(x=y, y=x)
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
# We will use the official tokenization script created by the Google team
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
# Load BERT from the Tensorflow Hub
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
import tokenization
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
# Encode the text into tokens, masks, and segment flags
train_input = bert_encode(df_train.text.values, tokenizer, max_len=160)
test_input = bert_encode(df_test.text.values, tokenizer, max_len=160)
train_labels = df_train.target.values
# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
# Build BERT model with my tuning
model_BERT = build_model(bert_layer, max_len=160)
model_BERT.summary()
# random_state_split = 32
# Dropout_num = 0
# learning_rate = 6e-6
# valid = 0.2
# epochs_num = 3
# batch_size_num = 16
# target_corrected = False
# target_big_corrected = False
checkpoint = ModelCheckpoint('model_BERT.h5', monitor='val_loss', save_best_only=True)

model_BERT.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint],
    batch_size=8
)
# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
# Prediction by BERT model with my tuning
model_BERT.load_weights('model_BERT.h5')
test_pred_BERT = model_BERT.predict(test_input)
test_pred_BERT_int = test_pred_BERT.round().astype('int')
# Prediction by BERT model with my tuning for the training data - for the Confusion Matrix
train_pred_BERT = model_BERT.predict(train_input)
train_pred_BERT_int = train_pred_BERT.round().astype('int')

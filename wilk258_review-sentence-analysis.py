# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf 
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
import re,string,unicodedata
from nltk.stem.porter import PorterStemmer
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
reviews = pd.read_csv("/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv")
reviews.head(5)
sns.countplot(x='overall', data=reviews)
feature = ['reviewText','overall','summary']
df = reviews[feature]
df.head(5)
df.shape
import spacy 
nlp = spacy.load('en_core_web_lg')
def text_entity(text):
    doc = nlp(text)
    for ent in doc.ents:
        print(f'Entity: {ent}, Label: {ent.label_}, {spacy.explain(ent.label_)}')

text_entity(df['reviewText'][10])
first = df['reviewText'][10]
doc = nlp(first)
spacy.displacy.render(doc, style='ent',jupyter=True)
second = df['reviewText'][150]
doc = nlp(second)
spacy.displacy.render(doc, style='ent',jupyter=True)
txt = df['reviewText'][10]
doc = nlp(txt)
spacy.displacy.render(doc, style='ent', jupyter=True)

for idx, sentence in enumerate(doc.sents):
    for noun in sentence.noun_chunks:
        print(f"sentence {idx+1} has noun chunk '{noun}'")
txt = df['reviewText'][10]
doc = nlp(txt)
spacy.displacy.render(doc, style='ent', jupyter=True)

for token in doc:
    print(token, token.pos_)
df_ = df['reviewText'].str.cat(sep=' ')

max_length = 1000000-1
df_ =  df_[:max_length]

import re
url_reg  = r'[a-z]*[:.]+\S+'
df_   = re.sub(url_reg, '', df_)
noise_reg = r'\&amp'
df_   = re.sub(noise_reg, '', df_)
doc = nlp(df_)
items_of_interest = list(doc.noun_chunks)
items_of_interest = [str(x) for x in items_of_interest]
df_nouns = pd.DataFrame(items_of_interest, columns=["instrument"])
plt.figure(figsize=(5,4))
sns.countplot(y="instrument",
             data=df_nouns,
             order=df_nouns["instrument"].value_counts().iloc[:10].index)
plt.show()
distri = df['reviewText'][200]
doc = nlp(distri)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
distri = df['reviewText'][500]
doc = nlp(distri)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])
for token in doc:
    print(f"token: {token.text},\t dep: {token.dep_},\t head: {token.head.text},\t pos: {token.head.pos_},\
    ,\t children: {[child for child in token.children]}")
stemmer = PorterStemmer()
def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text)    
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df.summary))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df.reviewText))
plt.imshow(wc , interpolation = 'bilinear')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras import layers
df.head(3)
def clean(text):
    text = text.fillna("fillna").str.lower()
    text = text.map(lambda x: re.sub('\\n',' ',str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))
 #   text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
 #   text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))
    return text

df["reviewText"] = clean(df["reviewText"])
df["summary"] = clean(df["summary"])
df.head(3)
def label(overall):
    if (overall == '1' or overall == '2' or overall == '3'):
        return 0
    else:
        return 1
df.overall = df.overall.apply(label) 
df.head(3)
vocab_size = 20000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 10000
sentences = df.reviewText.values
labels = df.overall.values
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
num_epochs = 50
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1)
def plot_result(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epoch")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_result(history, 'accuracy')
plot_result(history, 'loss')

model.evaluate(training_padded,training_labels)[1]

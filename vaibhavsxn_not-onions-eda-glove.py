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
#!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from collections import  Counter

plt.style.use('ggplot')

stop=set(stopwords.words('english'))

import re

from nltk.tokenize import word_tokenize

import gensim

import string

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
data = pd.read_csv('/kaggle/input/onion-or-not/OnionOrNot.csv')

print('Dataset Shape = {}'.format(data.shape))

print('Dataset Memory Usage = {:.2f} MB'.format(data.memory_usage().sum() / 1024**2))
data.head()
data.isna().sum()
x=data.label.value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

text_len=data[data['label']==1]['text'].str.len()

ax1.hist(text_len,color='red')

ax1.set_title('Onions')

text_len=data[data['label']==0]['text'].str.len()

ax2.hist(text_len,color='green')

ax2.set_title('Not Onion')

fig.suptitle('Characters in text')

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

text_len=data[data['label']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(text_len,color='red')

ax1.set_title('Onions')

text_len=data[data['label']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(text_len,color='green')

ax2.set_title('Not Onions')

fig.suptitle('Words in a text')

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

word=data[data['label']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')

ax1.set_title('Onions')

word=data[data['label']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')

ax2.set_title('Not Onions')

fig.suptitle('Average word length in each text')
def create_corpus(target):

    corpus=[]

    

    for x in data[data['label']==target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
corpus=create_corpus(0)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1

        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
x,y=zip(*top)

plt.bar(x,y)
corpus=create_corpus(1)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1



top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

    





x,y=zip(*top)

plt.bar(x,y)
plt.figure(figsize=(10,5))

corpus=create_corpus(1)



dic=defaultdict(int)

import string

special = string.punctuation

for i in (corpus):

    if i in special:

        dic[i]+=1

        

x,y=zip(*dic.items())

plt.bar(x,y)
plt.figure(figsize=(10,5))

corpus=create_corpus(0)



dic=defaultdict(int)

import string

special = string.punctuation

for i in (corpus):

    if i in special:

        dic[i]+=1

        

x,y=zip(*dic.items())

plt.bar(x,y,color='green')
counter=Counter(corpus)

most=counter.most_common()

x=[]

y=[]

for word,count in most[:40]:

    if (word not in stop) :

        x.append(word)

        y.append(count)
sns.barplot(x=y,y=x)
def get_top_text_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize=(10,5))

top_text_bigrams=get_top_text_bigrams(data['text'])[:10]

x,y=map(list,zip(*top_text_bigrams))

sns.barplot(x=y,y=x)
example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



remove_URL(example)
data['text']=data['text'].apply(lambda x : remove_URL(x))
example = """<div>

<h1>Onion or Not Onion</h1>

<p>Kaggle </p>

<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>

</div>"""
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

print(remove_html(example))
data['text']=data['text'].apply(lambda x : remove_html(x))
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



remove_emoji("Omg another sad day 😔😔")
data['text']=data['text'].apply(lambda x: remove_emoji(x))
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="I am a #king"

print(remove_punct(example))
data['text']=data['text'].apply(lambda x : remove_punct(x))
!pip install pyspellchecker
from spellchecker import SpellChecker

spell = SpellChecker()

def correct_spellings(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)

        

text = "corect me plese"

correct_spellings(text)
#data['text']=data['text'].apply(lambda x : correct_spellings(x)#)
def create_corpus(df):

    corpus=[]

    for tweet in tqdm(data['text']):

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

        corpus.append(words)

    return corpus
corpus=create_corpus(data)
#/kaggle/input/glove-vectors/glove.6B.100d.txt
embedding_dict={}

with open('/kaggle/input/glove-vectors/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
MAX_LEN=100

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences=tokenizer_obj.texts_to_sequences(corpus)



text_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))
num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec
model=Sequential()



embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=1e-5)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
model.summary()
train=text_pad[:data.shape[0]]

#test=text_pad[data.shape[0]:]
X_train,X_test,y_train,y_test=train_test_split(train,data['label'].values,test_size=0.15)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
model.fit(X_train,y_train,batch_size=16,epochs=15,validation_data=(X_test,y_test),verbose=2)
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
y_pred = model.predict(X_test)
y_pred = y_pred > 0.5
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_pred, y_test)
confusion_matrix(y_pred, y_test)
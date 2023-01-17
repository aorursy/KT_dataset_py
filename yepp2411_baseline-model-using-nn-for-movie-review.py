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
import zipfile



files=['/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip',

       '/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip',

       '/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip']



for file in files :

    zip = zipfile.ZipFile(file,'r')

    zip.extractall()

    zip.close()
train=pd.read_csv('/kaggle/working/labeledTrainData.tsv', delimiter="\t")

test=pd.read_csv('/kaggle/working/testData.tsv', delimiter="\t")
sub=pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/sampleSubmission.csv')
train.head()
print('the train data is : {} line'.format(len(train)))

print('the test data is : {} line'.format(len(test)))
train_len=train['review'].apply(len)

test_len=test['review'].apply(len)



import matplotlib.pyplot as plt

import seaborn as sns

fig=plt.figure(figsize=(15,4))

fig.add_subplot(1,2,1)

sns.distplot((train_len),color='red')



fig.add_subplot(1,2,2)

sns.distplot((test_len),color='blue')
train['word_n'] = train['review'].apply(lambda x : len(x.split(' ')))

test['word_n'] = test['review'].apply(lambda x : len(x.split(' ')))



fig=plt.figure(figsize=(15,4))

fig.add_subplot(1,2,1)

sns.distplot(train['word_n'],color='red')



fig.add_subplot(1,2,2)

sns.distplot(test['word_n'],color='blue')

train['length']=train['review'].apply(len)

train['length'].describe()
train['word_n'].describe()
from wordcloud import WordCloud

cloud=WordCloud(width=800, height=600).generate(" ".join(train['review'])) # join function can help merge all words into one string. " " means space can be a sep between words.

plt.figure(figsize=(15,10))

plt.imshow(cloud)

plt.axis('off')
fig, axe = plt.subplots(1,3, figsize=(23,5))

sns.countplot(train['sentiment'], ax=axe[0])

sns.boxenplot(x=train['sentiment'], y=train['length'], data=train, ax=axe[1])

sns.boxenplot(x=train['sentiment'], y=train['word_n'], data=train, ax=axe[2])
print('the review with question mark is {}'.format(np.mean(train['review'].apply(lambda x : '?' in x))))

print('the review with fullstop mark is {}'.format(np.mean(train['review'].apply(lambda x : '.' in x))))

print('the ratio of the first capital letter is {}'.format(np.mean(train['review'].apply(lambda x : x[0].isupper()))))

print('the ratio with the capital letter is {}'.format(np.mean(train['review'].apply(lambda x : max(y.isupper() for y in x)))))

print('the ratio with the number is {}'.format(np.mean(train['review'].apply(lambda x : max(y.isdigit() for y in x)))))
import re

import json

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer
train['review']=train['review'].apply(lambda x: BeautifulSoup(x,"html5lib").get_text())

test['review']=test['review'].apply(lambda x: BeautifulSoup(x,"html5lib").get_text())
train['review']=train['review'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))

test['review']=test['review'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))
train.head(3)
stops = set(stopwords.words("english"))



for i in range(0,25000) : 

    review = train.iloc[i,2] # review column : 2 

    review = review.lower().split()

    words = [r for r in review if not r in stops]

    clean_review = ' '.join(words)

    train.iloc[i,2] = clean_review
for i in range(0,25000) : 

    review = test.iloc[i,1] # review column : 1

    review = review.lower().split()

    words = [r for r in review if not r in stops]

    clean_review = ' '.join(words)

    test.iloc[i,1] = clean_review
train['word_n_2'] = train['review'].apply(lambda x : len(x.split(' ')))

test['word_n_2'] = test['review'].apply(lambda x : len(x.split(' ')))



fig, axe = plt.subplots(1,1, figsize=(7,5))

sns.boxenplot(x=train['sentiment'], y=train['word_n_2'], data=train)
from keras.preprocessing.text import Tokenizer

tk = Tokenizer()

tk.fit_on_texts(list(train['review'])+list(test['review']))

text_seq_tr=tk.texts_to_sequences(train['review'])

text_seq_te=tk.texts_to_sequences(test['review'])

word_ind=tk.word_index
print('Total word count is :',len(word_ind))
data_info={}

data_info['word_ind']=word_ind

data_info['word_len']=len(word_ind)+1
import matplotlib.pyplot as plt

import seaborn as sns



fig=plt.figure(figsize=(15,4))

fig.add_subplot(1,2,1)

sns.distplot(pd.Series(text_seq_tr).apply(lambda x : len(x)))

fig.add_subplot(1,2,2)

sns.distplot(pd.Series(text_seq_te).apply(lambda x : len(x)))
from keras.preprocessing.sequence import pad_sequences

pad_train=pad_sequences(text_seq_tr, maxlen=400) 

pad_test=pad_sequences(text_seq_te, maxlen=400) 
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(pad_train, train['sentiment'], random_state=77, test_size=0.07, stratify=train['sentiment'])
len(tk.word_index)
from keras import Sequential

from keras.layers import Dense, Embedding, Flatten



model=Sequential()

model.add(Embedding(101247,65, input_length=400))

model.add(Flatten())

model.add(Dense(2,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'] )
from keras.callbacks import EarlyStopping, ModelCheckpoint

es=EarlyStopping(patience=4) 

mc=ModelCheckpoint('best.h5',save_best_only=True)

model.fit(x_train,y_train, batch_size=128, epochs=10, validation_data=[x_valid,y_valid], callbacks=[es,mc]) 
model.load_weights('best.h5')
res=model.predict(pad_test, batch_size=128)
res
sub['sentiment_pro']=res[:,1]
sub.loc[sub['sentiment_pro']>=0.5,"sentiment"]=1

sub.loc[sub['sentiment_pro']<0.5,"sentiment"]=0
sub=sub[['id','sentiment']]
sub.to_csv('result.csv',index=False)
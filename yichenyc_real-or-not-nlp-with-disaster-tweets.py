# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



from pandas import DataFrame

import string

from wordcloud import STOPWORDS

import re

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from tqdm import tqdm

from nltk.tokenize import word_tokenize

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.initializers import Constant

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/nlp-getting-started/train.csv')

data_test = pd.read_csv('../input/nlp-getting-started/test.csv')



print('Training data shape: ', format(data_train.shape))

print('Test data shape: ', format(data_test.shape))



data_train.head(3)
missing_checkCols = ['id', 'keyword', 'location', 'text']



missingNumber_train = []

missingPercent_train = []

missingNumber_test = []

missingPercent_test = []



for col in missing_checkCols:

    missingNumber_train.append(data_train[col].isna().sum())

    missingPercent_train.append(data_train[col].isna().sum()/data_train[col].count())

    missingNumber_test.append(data_test[col].isna().sum())

    missingPercent_test.append(data_test[col].isna().sum()/data_test[col].count())

    



missing = {'Variable': missing_checkCols,

            'Number in Train':  missingNumber_train, 

            'Number in Test': missingNumber_test, 

            'Percent in Train':   missingPercent_train, 

            'Percent in Test':  missingPercent_test}



missingFrame = DataFrame(missing)

missingFrame

DISASTER_ITEM = data_train['target'] == 1

FEATURES = ['Character_Number', 'Word_Number', 'Stopword_Number', 'Punctuation_Number', 'URL_Number']



data_train['Character_Number'] = data_train['text'].str.len()

data_train['Word_Number'] = data_train['text'].str.split().map(lambda x:len(x))

data_train['Stopword_Number'] = data_train['text'].apply(lambda x: len([y for y in str(x).lower().split() if y in STOPWORDS]))

data_train['Punctuation_Number'] = data_train['text'].apply(lambda x: len([y for y in str(x) if y in string.punctuation]))

data_train['URL_Number'] = data_train['text'].apply(lambda x: len([y for y in str(x).lower().split() if 'http' in y or 'https' in y]))



fig, axes = plt.subplots(ncols = 1, nrows = len(FEATURES), figsize=(5, 25), dpi = 100)



for i, feature in enumerate(FEATURES):

    sns.distplot(data_train.loc[DISASTER_ITEM][feature], label = 'Disaster', ax = axes[i])

    sns.distplot(data_train.loc[~DISASTER_ITEM][feature], label = 'Not Disaster', ax = axes[i])

    

    axes[i].legend()

    axes[i].set_title(f'{feature} Distribution in Training Data', fontsize = 14)

    



plt.show()
def remove_url(txt):

    txt = re.sub(r'https?://\S+|www\.\S+', "", txt)

    return txt



data_train['cleaned'] = data_train['text'].apply(lambda x: remove_url(x))

data_test['cleaned'] = data_test['text'].apply(lambda x: remove_url(x))
def remove_punctuation(txt):

    table=str.maketrans('','',string.punctuation)

    return txt.translate(table)



data_train['cleaned'] = data_train['cleaned'].apply(lambda x: remove_punctuation(x))

data_test['cleaned'] = data_test['cleaned'].apply(lambda x: remove_punctuation(x))
def create_corpus(data):

    corpus=[]

    for tweet in tqdm(data['cleaned']):

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in STOPWORDS))]

        corpus.append(words)

    return corpus



corpus_train = create_corpus(data_train)

corpus_test = create_corpus(data_test)



embedding_dict={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
MAX_LEN=50



tokenizer_obj_train=Tokenizer()

tokenizer_obj_train.fit_on_texts(corpus_train)

sequences_train=tokenizer_obj_train.texts_to_sequences(corpus_train)

tweet_pad_train=pad_sequences(sequences_train,maxlen=MAX_LEN,truncating='post',padding='post')



tokenizer_obj_test=Tokenizer()

tokenizer_obj_test.fit_on_texts(corpus_test)

sequences_test=tokenizer_obj_test.texts_to_sequences(corpus_test)

tweet_pad_test=pad_sequences(sequences_test,maxlen=MAX_LEN,truncating='post',padding='post')



word_index_train=tokenizer_obj_train.word_index

word_index_test=tokenizer_obj_test.word_index
num_words_train=len(word_index_train)+1

embedding_matrix_train=np.zeros((num_words_train,100))



for word,i in tqdm(word_index_train.items()):

    if i > num_words_train:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix_train[i]=emb_vec

        



num_words_test=len(word_index_test)+1

embedding_matrix_test=np.zeros((num_words_test,100))



for word,i in tqdm(word_index_test.items()):

    if i > num_words_test:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix_test[i]=emb_vec
model=Sequential()



embedding=Embedding(num_words_train,100,embeddings_initializer=Constant(embedding_matrix_train),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=1e-5)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])



model.summary()
X_train,X_test,y_train,y_test=train_test_split(tweet_pad_train,data_train['target'].values,test_size=0.15)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)
sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
y_pre=model.predict(tweet_pad_test)

y_pre=np.round(y_pre).astype(int).reshape(3263)

sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pre})

sub.to_csv('submission.csv',index=False)
sub.head()
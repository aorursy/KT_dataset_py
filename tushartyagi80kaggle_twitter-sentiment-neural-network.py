import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, RNN

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers
#Importing training and test data

train = pd.read_csv("../input/twitter-sentiment-analysis-hatred-speech/train.csv")

test = pd.read_csv("..//input/twitter-sentiment-analysis-hatred-speech/test.csv")

print("Train shape : ",train.shape)

print("Test shape : ",test.shape)
# ------------Step 1 - Definig cleaning functions - URLs, Mentions, Negation handling, UF8 (BOM), Special chracters and numbers

#!pip install bs4

#!pip install nltk

#!pip install et_xmlfile



#!pip install lxml

import re

from bs4 import BeautifulSoup

from nltk.tokenize import WordPunctTokenizer

Tokenz = WordPunctTokenizer()

Mentions_Removal = r'@[A-Za-z0-9_]+'

Http_Removal = r'http(s?)://[^ ]+'

#HttpS_Removal = r'https://[^ ]+'

Www_Removal = r'www.[^ ]+'



#Combining the above 3 removals functions

#Combining_MentnHttp = r'|'.join((Mentions_Removal,Http_Removal))

Combining_MentnHttp1 = r'|'.join((Http_Removal,Www_Removal))





#Creating a negation dictionary because words with apostrophe symbol (') will (Can't > can t) 

Negation_Dictonary = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", 

                "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", 

                "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", 

                "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", 

                "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",

                "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 

                "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", 

                "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",

                "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 

                "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 

                "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 

                "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", 

                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",

                "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",

                "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",

                "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",

                "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 

                "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",

                "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  

                "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 

                "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", 

                "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", 

                "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 

                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",

                 "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

Negation_Joining= re.compile(r'\b(' + '|'.join(Negation_Dictonary.keys()) + r')\b')



def Clean_Question_Function(text):

    BeautifulSoup_assign = BeautifulSoup(text, 'html.parser')

    Souping = BeautifulSoup_assign.get_text()

    try:

        BOM_removal = Souping.decode("utf-8-sig").replace(u"\ufffd", "?")

    except:

        BOM_removal = Souping

    Comb_2 = re.sub(Combining_MentnHttp1, '', BOM_removal)

    #Comb_3 = re.sub(Www_Removal,'',Comb_2)

    Comb_3 = re.sub(Mentions_Removal,'',Comb_2)

    LowerCase = Comb_3.lower()

    Negation_Handling = Negation_Joining.sub(lambda x: Negation_Dictonary[x.group()], LowerCase)

    Letters_only = re.sub("[^a-zA-Z]", " ", Negation_Handling)

    

    # Removing unneccessary white- Tokenizing and joining together

    Tokenization = [x for x  in Tokenz.tokenize(Letters_only) if len(x) > 1]

    return (" ".join(Tokenization)).strip()

Clean_Question_Function



#Removing stop words from training and test

from nltk.corpus import stopwords

stopwords = set(stopwords.words('english')) - {'no', 'nor', 'not'}

def remove_stopwords(text):

    return ' '.join([word for word in str(text).split() if word not in stopwords])



# Lemmatization

from nltk.stem import WordNetLemmatizer

def get_lemmatized_text(corpus):

    lemmatizer = WordNetLemmatizer()

    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]



#Cleaning up the data with step 1

xrange = range #Defining X range
train.head()
test.head()
%%time

xrange = range

print ("Cleaning the reviews...\n")

clean_review_texts = []

for i in xrange(0,len(test)):

    if( (i+1)%100000 == 0 ):

        "Reviews %d of %d has been processed".format( i+1, len(test) )  

        

    clean_review_texts.append(Clean_Question_Function(test['tweet'][i]))

    

#Changing into dataframe

test['cleaned_tweet'] = clean_review_texts
%%time

xrange = range

print ("Cleaning the reviews...\n")

clean_review_texts = []

for i in xrange(0,len(train)):

    if( (i+1)%100000 == 0 ):

        "Reviews %d of %d has been processed".format( i+1, len(train) )  

        

    clean_review_texts.append(Clean_Question_Function(train['tweet'][i]))

    

#Changing into dataframe

train['cleaned_tweet'] = clean_review_texts
train.head()
#Stopwords

import string

from nltk.corpus import stopwords

eng_stopwords = set(stopwords.words("english"))
#Lemmatization

train['cleaned_tweet'] = get_lemmatized_text(train['cleaned_tweet'])

test['cleaned_tweet'] = get_lemmatized_text(test['cleaned_tweet'])
#Number of words

train['cleaned_tweet_len'] = train['cleaned_tweet'].str.len()

test['cleaned_tweet_len'] = test['cleaned_tweet'].str.len()



## Number of unique words in the text ##

train["num_unique_words"] = train["cleaned_tweet"].apply(lambda x: len(set(str(x).split())))

test["num_unique_words"] = test["cleaned_tweet"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train["num_chars"] = train["cleaned_tweet"].apply(lambda x: len(str(x)))

test["num_chars"] = test["cleaned_tweet"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

train["num_stopwords"] = train["cleaned_tweet"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

test["num_stopwords"] = test["cleaned_tweet"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))



## Number of punctuations in the text ##

train["num_punctuations"] =train["tweet"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test["num_punctuations"] =test["tweet"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of upper case words in the text ##

train["num_words_upper"] = train["tweet"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test["num_words_upper"] = test["tweet"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



## Number of title case words in the text ##

train["num_words_title"] = train["tweet"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

test["num_words_title"] = test["tweet"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



## Average length of the words in the text ##

train["mean_word_len"] = train["cleaned_tweet"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test["mean_word_len"] = test["cleaned_tweet"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
train.groupby('label').mean()
#Stop words

#train['cleaned_tweet'] = train['cleaned_tweet'].apply(lambda text: remove_stopwords(text))

#test['cleaned_tweet'] = test['cleaned_tweet'].apply(lambda text: remove_stopwords(text))
train = train.sample(frac=1)

train = train.reset_index(drop=True)

train.head()
test = test.sample(frac=1)

test = test.reset_index(drop=True)

test.head()
#train['cleaned_tweet'] = train['cleaned_tweet'].fillna("_##_").values

#test['cleaned_tweet'] = test['cleaned_tweet'].fillna("_##_").values
#Parameters for models

embed_size = 300 # how big is each word vector

max_features = 5000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 80 # max number of words in a question to use

SEED = 1029



## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features, split=' ')

tokenizer.fit_on_texts(train['cleaned_tweet'].values)



train_data = tokenizer.texts_to_sequences(train['cleaned_tweet'].values)

test_data = tokenizer.texts_to_sequences(test['cleaned_tweet'].values)



## Pad the sentences 

train_data_pad  = pad_sequences(train_data , maxlen=maxlen)

test_data_pad  = pad_sequences(test_data , maxlen=maxlen)
print("\nExamples:")

print(train['cleaned_tweet'][100], '-->', train_data[100])

print(train['cleaned_tweet'][200], '-->', train_data[200])

print(train['cleaned_tweet'][300], '-->', train_data[300])
from keras.layers import Input

def fit(self, X_train, y_train):

    self.model.fit(self.preprocessing(X_train), y_train, epochs=self.epochs, batch_size=512)



def guess(self, features):

        features = self.preprocessing(features)

        result = self.model.predict(features).flatten()

        return result
#Importing Glove and creating vector

EMBEDDING_FILE_GLOVE = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE_GLOVE))

all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()



embed_size = all_embs.shape[1]

word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix_glove = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix_glove[i] = embedding_vector

EMBEDDING_FILE_PARA = '../input/quora-insincere-questions-classification/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE_PARA, encoding="utf8", errors='ignore') if len(o)>80)



all_embs = np.stack(embeddings_index.values())



emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



# word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix_para = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix_para[i] = embedding_vector
EMBEDDING_FILE_W2V = '../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE_W2V) if len(o)>80)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix_w2v = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix_w2v[i] = embedding_vector
#Concatination Glove, W2V, and Para

embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_para,embedding_matrix_w2v), axis=1)
print("\nExample")

print(train_data[100], '-->', train_data_pad[100])
print('\nInput train data shape:', train_data_pad.shape)

print('Input test data shape:', test_data_pad.shape)
#Setting up train and test

train_X = train.drop(['id','tweet','num_unique_words','num_chars','num_words_upper','num_punctuations','mean_word_len','label','cleaned_tweet'],axis=1)#Dec

y = train['label'].values



test_X = test.drop(['id','tweet','num_unique_words','num_chars','num_words_upper','num_punctuations','mean_word_len','cleaned_tweet'],axis=1)



print(train_X.columns)

print(test_X.columns)

print(train_X.shape)

print(test_X.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(train_X.values)

scaled_train_X = scaler.transform(train_X)

scaled_test_X = scaler.transform(test_X)
def split_features(X):

    

    X_list = []

    x_0 = train_data_pad[..., :]

    X_list.append(x_0)

    

    x_1 = X[..., :]

    X_list.append(x_1)



    return X_list
embed_size
from keras.layers import Input, Dense, Dropout, Conv1D, Embedding, SpatialDropout1D, Concatenate,GRU, LSTM,Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D,CuDNNLSTM, CuDNNGRU, Activation, Reshape

inp1 = Input(shape=(maxlen,))

t = Embedding(max_features, embed_size*3, weights=[embedding_matrix])(inp1)

t = SpatialDropout1D(0.1)(t)

t1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(t)

t2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(t)

max_pool1 = GlobalMaxPooling1D()(t1)

max_pool2 = GlobalMaxPooling1D()(t2)

conc = Concatenate()([max_pool1, max_pool2])

out_1 = Dense(32, activation="relu")(conc)







inp_2 = Input(shape=(3,))

dense_2 = Dense(120,activation='relu')(inp_2)

out_dense_2 = Reshape(target_shape=(120,))(dense_2)

dense_3 = Dense(32,activation='relu')(out_dense_2)

out_dense_3 = Reshape(target_shape=(32,))(dense_3)



input_model = [inp1, inp_2]

output_model = [out_1, out_dense_3]



output = Concatenate()(output_model)

output = Dense(16, activation='relu')(output)

output = Dropout(0.2)(output)

output = Dense(1, activation='sigmoid')(output)



adam = optimizers.Adam(lr=0.007)

modelll = Model(inputs=input_model, outputs=output)

modelll.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

print(modelll.summary())
## Train the model 

modelll.fit(split_features(train_X.values), y, batch_size=512, epochs=50,shuffle=True,validation_split=0.20)
def split_features(X):

    

    X_list = []

    x_0 = test_data_pad[..., :]

    X_list.append(x_0)

    

    x_1 = X[..., :]

    X_list.append(x_1)



    return X_list
#Predicting on test set

pred32 = modelll.predict(split_features(test_X.values))

#Checking the accuracy

#pred_test = modelll.predict([test_X],batch_size=512, verbose=1)

pred_test_y = (pred32>0.5).astype(int)

out_df = pd.DataFrame({"id":test["id"].values})

out_df['label'] = pred_test_y

out_df.to_csv("submission_wikinpara.csv", index=False)
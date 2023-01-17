import json

import pandas as pd

from gensim.models import word2vec

from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import itertools

import numpy as np

from keras.layers import Input,Embedding,SpatialDropout1D,Bidirectional,LSTM,CuDNNLSTM,CuDNNGRU,GlobalAveragePooling1D,GlobalMaxPooling1D,Dense,Dropout,concatenate

from keras.models import Model

from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

import operator

import re

from keras import backend as K

import gc



import warnings

warnings.filterwarnings('ignore')
train_df=pd.read_csv('../input/nlp-getting-started/train.csv')

print('train_df.shape',train_df.shape)

train_df.head()
def get_vocab(sentences):

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab

sentences=train_df['text'].apply(lambda x:x.split())

vocab=get_vocab(sentences.values)

sorted(vocab.items(), key=operator.itemgetter(1),reverse = True)[:10]
news_path = '../input/embedding/GoogleNews-vectors-negative300.bin'

embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
def check_coverage(vocab,embeddings_index):

    counts=0

    oov={}

    all_counts=sum(vocab.values())

    

    intersection=list(vocab.keys() & embeddings_index.vocab.keys())

    difference=list(vocab.keys() - embeddings_index.vocab.keys()) #words in vocab but not in embeddings_index

    for key in intersection:

        counts+=vocab[key]

    for key in difference:

        oov[key]=vocab[key]

    print('embedding_word coverage: {:.2%}'.format(len(intersection) / len(vocab)))

    print('embedding_counts coverage: {:.2%}'.format(counts / all_counts))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1),reverse=True)



    return sorted_x,difference
oov,difference=check_coverage(vocab,embeddings_index)

print('the nums of words which are in vocab not in embed:',len(difference))

oov[:20]


def clean_text(x):

    x = str(x)

    x=x.replace("'s", ' is')

    x=x.replace("'t", ' not') 

    x=x.replace('\x89', '')

    for punct in '&?!.,#$%()*+-/:;<=>@[\\]^_`{|}~'+"'":

        x=x.replace(punct, '')

    x=re.sub('[0-9]','',x)  #if numberic is not embedding

    text = re.sub('https?://\S+|www\.\S+', '', x)

    return x
sentences=train_df['text'].apply(lambda x:clean_text(x).split())

vocab=get_vocab(sentences.values)

sorted(vocab.items(), key=operator.itemgetter(1),reverse = True)[:10]

oov,difference=check_coverage(vocab,embeddings_index)

print('the nums of words which are in vocab not in embed:',len(difference))

oov[:50]
print("'traumatise' in embeddings_index",'traumatise' in embeddings_index)

print("'pre-break' in embeddings_index",'pre-break' in embeddings_index)

print("'disease' in embeddings_index",'disease' in embeddings_index)#Actually ,when I look at the csv data ,I find that 'disea' is 'disea...'

print("'Typhoon' in embeddings_index",'Typhoon' in embeddings_index)

print("'Devastated' in embeddings_index",'Devastated' in embeddings_index)

print("'sub-reddits' in embeddings_index",'sub-reddits' in embeddings_index)

print("'neighbor' in embeddings_index",'neighbor' in embeddings_index)

print("'realize' in embeddings_index",'realize' in embeddings_index)



def replace_word(x):

    x=x.replace('traumatised','traumatise')

    x=x.replace('colour','color')

    x=x.replace('disea.','disease')

    x=x.replace('TyphoonDevastated','Typhoon Devastated')

    x=x.replace('neighbour','neighbor')

    x=x.replace('realise','realize')

    return x
sentences=train_df['text'].apply(lambda x:replace_word(x)).apply(lambda x:clean_text(x).split())

vocab=get_vocab(sentences.values)

sorted(vocab.items(), key=operator.itemgetter(1),reverse = True)[:10]
oov,difference=check_coverage(vocab,embeddings_index)

print('the nums of words which are in vocab not in embed:',len(difference))

oov[:20]
## Tokenize the sentences

train_y=train_df['target']

tokenizer = Tokenizer()

tokenizer.fit_on_texts(list(sentences))

train_sequence = tokenizer.texts_to_sequences(sentences)



## Pad the sentences 

pad_length=int(len(list(itertools.chain(*train_sequence)))/len(train_sequence))

print('pad_length',pad_length)

train_pad=pad_sequences(train_sequence,pad_length)



print('train_pad.shape',train_pad.shape)

train_pad[:5]
#######get embed_matrix########

all_embs = embeddings_index.vectors

print('all_embs.shape',all_embs.shape)

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

embedding_matrix = np.random.normal(emb_mean, emb_std, (len(word_index), embed_size))

for word, i in word_index.items():

    if word in embeddings_index:

        embedding_matrix[i] = embeddings_index[word]



embedding_matrix.shape
##################model#######################

def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def model_lstm(embedding_matrix):

    

    inp = Input(shape=(pad_length,))

    embedding = Embedding(len(word_index), embed_size, weights=[embedding_matrix], trainable=False)(inp)

    dropout1 = SpatialDropout1D(0.1)(embedding)

    lstm = Bidirectional(LSTM(40, return_sequences=True))(dropout1)



    avg_pool = GlobalAveragePooling1D()(lstm)

    max_pool = GlobalMaxPooling1D()(lstm)



    conc=concatenate([avg_pool, max_pool])

    dense1 = Dense(16, activation="relu")(conc)

    dropout2 = Dropout(0.1)(dense1)

    outp = Dense(1, activation="sigmoid")(dropout2)    



    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[f1])   

    return model
def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in [i * 0.05 for i in range(20)]:

        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result



def train_pred(model, train_X, train_y, val_X, val_y,test_X, epochs=2):

    model.fit(train_X, train_y, batch_size=200, epochs=epochs, validation_data=(val_X, val_y), verbose=0)

    pred_val_y = model.predict([val_X], batch_size=200, verbose=0)



    search_result=threshold_search(val_y,pred_val_y)

    print('best search_result:',search_result)



    pred_test_y = model.predict([test_X], batch_size=200, verbose=0)

    

    return pred_val_y, pred_test_y

test_df=pd.read_csv('../input/nlp-getting-started/test.csv')

sentences_test=test_df['text'].apply(lambda x:replace_word(x)).apply(lambda x:clean_text(x).split())



test_sequence = tokenizer.texts_to_sequences(sentences_test)

test_pad=pad_sequences(test_sequence,pad_length)
train_all = np.zeros(train_y.shape)

test_all = np.zeros(test_pad.shape[0])

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=2020).split(train_pad, train_y))

for idx, (train_idx, valid_idx) in enumerate(splits):

    print('###########valid {}##########'.format(idx))

    X_train = train_pad[train_idx]

    y_train = train_y[train_idx]

    X_val = train_pad[valid_idx]

    y_val = train_y[valid_idx]

    model = model_lstm(embedding_matrix)

    pred_val_y, pred_test_y = train_pred(model, X_train, y_train, X_val, y_val,test_pad, epochs =100)

    train_all[valid_idx] = pred_val_y.reshape(-1)

    test_all += pred_test_y.reshape(-1) / len(splits)
sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

sub["target"]= (test_meta > 0.33).astype(int)

sub.to_csv("submission.csv", index=False)
import collections

collections.Counter(sub['target'])
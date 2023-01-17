import numpy as np # linear algebra
np.set_printoptions(threshold=np.nan)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/embeddings"))
print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300"))

# Any results you write to the current directory are saved as output.

import gensim
from gensim.utils import simple_preprocess
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score,precision_recall_fscore_support,recall_score,precision_score
from keras import backend as K
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

#https://www.kaggle.com/shujian/single-rnn-with-4-folds-v1-9
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        print('\rthreshold = %f | score = %f'%(threshold,score),end='')
        if score > best_score:
            best_threshold = threshold
            best_score = score
    print('\nbest threshold is % f with score %f'%(best_threshold,best_score))
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result
df = pd.read_csv('../input/train.csv')
df["question_text"].fillna("_##_",inplace=True)
max_len = df['question_text'].apply(lambda x:len(x)).max()
print('max length of sequences:',max_len)
# df = df.sample(frac=0.1)

print('columns:',df.columns)
pd.set_option('display.max_columns',None)
print('df head:',df.head())
print('example of the question text values:',df['question_text'].head().values)
print('what values contains target:',df.target.unique())

print('Computing class weights....')
#https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(df.target.values),
                                                 df.target.values)
print('class_weights:',class_weights)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#dim of vectors
dim = 300
# max words in vocab
num_words = 50000
# max number in questions
max_len = 100 

print('Fiting tokenizer')
## Tokenize the sentences
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(df['question_text'])

print('spliting data')
df_train,df_test = train_test_split(df, random_state=1)

print('text to sequence')
x_train = tokenizer.texts_to_sequences(df_train['question_text'])
x_test = tokenizer.texts_to_sequences(df_test['question_text'])

print('pad sequence')
## Pad the sentences 
x_train = pad_sequences(x_train,maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

## Get the target values
y_train = df_train['target'].values
y_test = df_test['target'].values

print(x_train.shape)
print(y_train.shape)

# https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
print('loading word2vec model...')
word2vec = gensim.models.KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
print('vocab:',len(word2vec.vocab))

all_embs = word2vec.vectors
emb_mean,emb_std = all_embs.mean(), all_embs.std()
print(emb_mean,emb_std)

print(num_words,' from ',len(tokenizer.word_index.items()))
# num_words = min(num_words, len(tokenizer.word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, dim))

# embedding_matrix = np.zeros((num_words, dim))
count = 0
for word, i in tokenizer.word_index.items():
    if i>=num_words:
        break
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
    else:
        count += 1
print('embedding matrix size:',embedding_matrix.shape)
print('Number of words not in vocab:',count)

del word2vec
import gc
gc.collect()
from keras.layers import Dense, Input,Embedding, Dropout, Activation, CuDNNLSTM,BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, Concatenate, GlobalAveragePooling1D,Average,SpatialDropout1D
from keras.layers.noise import AlphaDropout
from keras.models import Model
from keras.callbacks import Callback,EarlyStopping
from keras.engine import Layer
from keras.initializers import Ones, Zeros
import keras.backend as K
from keras import regularizers
from keras import constraints
from keras import optimizers

# https://ai.google/research/pubs/pub46697
adam = optimizers.Adam()
print('LR:',K.eval(adam.lr))
# 0.001 = learning rate in adam
# optimal batch size ~ eps *N, where eps = learning rate and N = training size
batch_size = int(x_train.shape[0]*K.eval(adam.lr))
print('Batch size = ',batch_size)


#model looks to be from here: https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-069

#word2vec    
inp1 = Input(shape=(max_len,))
e1 = Embedding(num_words, dim, weights=[embedding_matrix], trainable=True,embeddings_regularizer=regularizers.l2(0.0001))(inp1)
e1 = SpatialDropout1D(0.1)(e1)
e1 = Bidirectional(CuDNNLSTM(50,return_sequences=True,kernel_regularizer=regularizers.l2(0.0001)))(e1)
e1 = GlobalMaxPool1D()(e1)
e1 = BatchNormalization()(e1)
e1 = Dropout(0.1)(e1)
e1 = Dense(50, activation="selu",kernel_regularizer=regularizers.l2(0.0001))(e1)
e1 = BatchNormalization()(e1)
e1 = AlphaDropout(0.4)(e1)
x = Dense(1, activation="sigmoid",kernel_regularizer=regularizers.l2(0.0001))(e1)

model = Model(inputs=inp1, outputs=x)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

print(model.summary())

# for commiting the model to competition i need to comment these sections....otherwise the running time will be more then 2h on gpu...
history = model.fit(x_train,y_train, 
                      batch_size=batch_size, 
                      validation_split=0.33,
                      epochs=100,
                      #overfits rather soon
                      callbacks=[EarlyStopping(patience=2)])

print('training done....')
_, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(history.history['loss'], label='loss')
ax[0].plot(history.history['val_loss'], label='val_loss')
ax[0].legend()
ax[0].set_title('loss')

ax[1].plot(history.history['acc'], label='acc')
ax[1].plot(history.history['val_acc'], label='val_acc')
ax[1].legend()
ax[1].set_title('acc')

plt.show()
#for train set
y_pred = model.predict(x_train,batch_size=1024, verbose=1)
search_result = threshold_search(y_train, y_pred)
print(search_result)
y_pred = y_pred>search_result['threshold']
y_pred = y_pred.astype(int)

print('RESULTS ON TRAINING SET:\n',classification_report(y_train,y_pred))


#for test set
y_pred = model.predict(x_test,batch_size=1024, verbose=1)
search_result = threshold_search(y_test, y_pred)
print(search_result)
y_pred = y_pred>search_result['threshold']
y_pred = y_pred.astype(int)

print('RESULTS ON TEST SET:\n',classification_report(y_test,y_pred))
#fit final model on all data
print('text to sequence')
x = tokenizer.texts_to_sequences(df['question_text'])

print('pad sequence')
## Pad the sentences 
x = pad_sequences(x,maxlen=max_len)

## Get the target values
y = df['target'].values

print('fiting final model...')
history = model.fit(x,y, batch_size=batch_size, epochs=8)

print('fitting on full data done...')
_, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(history.history['loss'], label='loss')
ax[0].legend()
ax[0].set_title('loss')

ax[1].plot(history.history['acc'], label='acc')
ax[1].legend()
ax[1].set_title('acc')

plt.show()

y_pred = model.predict(x,batch_size=batch_size, verbose=1)
search_result = threshold_search(y, y_pred)
y_pred = y_pred>search_result['threshold']
y_pred = y_pred.astype(int)

print(classification_report(y,y_pred))
#submission
print('Loading test data...')
df_final = pd.read_csv('../input/test.csv')
df_final["question_text"].fillna("_##_", inplace=True)

x_final=tokenizer.texts_to_sequences(df_final['question_text'])
x_final = pad_sequences(x_final,maxlen=max_len)

y_pred = model.predict(x_final,batch_size=batch_size,verbose=1)
y_pred = y_pred > search_result['threshold']
y_pred = y_pred.astype(int)
print(y_pred[:5])

df_subm = pd.DataFrame()
df_subm['qid'] = df_final.qid
df_subm['prediction'] = y_pred
print(df_subm.head())
df_subm.to_csv('submission.csv', index=False)
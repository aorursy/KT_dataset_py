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
print('glove...')

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt'))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
print(len(all_embs))


word_index = tokenizer.word_index
# num_words = min(num_words, len(word_index))
embedding_matrix_glov = np.random.normal(emb_mean, emb_std, (num_words, dim))
count=0
for word, i in word_index.items():
    if i >= num_words: 
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix_glov[i] = embedding_vector
    else:
        count += 1
print('embedding matrix size:',embedding_matrix_glov.shape)
print('Number of words not in vocab:',count)

del embeddings_index,all_embs
import gc
gc.collect()

print('paragram...')

EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
print(len(all_embs))


word_index = tokenizer.word_index
# num_words = min(num_words, len(word_index))
embedding_matrix_para = np.random.normal(emb_mean, emb_std, (num_words, dim))
count=0
for word, i in word_index.items():
    if i >= num_words: 
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix_para[i] = embedding_vector
    else:
        count += 1
print('embedding matrix size:',embedding_matrix_glov.shape)
print('Number of words not in vocab:',count)

del embeddings_index,all_embs
import gc
gc.collect()

print('wiki news...')

EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

print(len(all_embs))

word_index = tokenizer.word_index
embedding_matrix_wiki = np.random.normal(emb_mean, emb_std, (num_words, dim))

count=0
for word, i in word_index.items():
    if i >= num_words: 
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix_wiki[i] = embedding_vector
    else:
        count += 1
print('embedding matrix size:',embedding_matrix_wiki.shape)
print('Number of words not in vocab:',count)

del embeddings_index,all_embs
import gc
gc.collect()

from keras.layers import Dense, Input,Embedding, Dropout, Activation, CuDNNLSTM,BatchNormalization,concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D, Concatenate, GlobalAveragePooling1D,Average,Conv1D,GlobalMaxPooling1D
from keras.models import Model
from keras.callbacks import Callback,EarlyStopping
from keras.engine import Layer
from keras.initializers import Ones, Zeros
import keras.backend as K
from keras import regularizers
from keras import constraints

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

# https://arxiv.org/abs/1607.06450
# https://github.com/keras-team/keras/issues/3878
class LayerNormalization(Layer):
    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gain = self.add_weight(name='gain', shape=input_shape[-1:],
                                    initializer=Ones(), trainable=True)
        self.bias = self.add_weight(name='bias', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, **kwargs):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        # dot = *
        # std+eps because of possible nans..
        return self.gain * (x - mean) / (std + K.epsilon()) + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

# Yoon Kim's shallow cnn model: https://arxiv.org/pdf/1408.5882.pdf
# https://github.com/raghakot/keras-text/blob/master/keras_text/models/sequence_encoders.py

#word2vec    
inp1 = Input(shape=(max_len,))
e1 = Embedding(num_words, dim, weights=[embedding_matrix])(inp1)

#glove
inp2 = Input(shape=(max_len,))
e2 = Embedding(num_words, dim, weights=[embedding_matrix_glov])(inp2)

#wiki
inp3 = Input(shape=(max_len,))
e3 = Embedding(num_words, dim, weights=[embedding_matrix_wiki])(inp3)

#para
inp4 = Input(shape=(max_len,))
e4 = Embedding(num_words, dim, weights=[embedding_matrix_para])(inp4)

#no pretriained
inp5 = Input(shape=(max_len,))
e5 = Embedding(num_words, dim,)(inp5)

x = Concatenate()([e1,e2,e3,e4,e5])
pooled_tensors = []
kernel_sizes = [3, 4, 5]
for kernel_size in kernel_sizes:
    l1 = Conv1D(100, kernel_size=kernel_size, activation='relu', )(x)
    l1 = GlobalMaxPooling1D()(l1)
    pooled_tensors.append(l1)

if len(kernel_sizes) > 1:
    x = concatenate(pooled_tensors, axis=-1)
else:
    x = pooled_tensors[0]

x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[inp1,inp2, inp3, inp4,inp5], outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# clr = CyclicLR(base_lr=0.001, max_lr=0.007,
#                         step_size=300., mode='exp_range',
#                         gamma=0.99994)
# clr = CyclicLR()


# # for commiting the model to competition i need to comment these sections....otherwise the running time will be more then 2h on gpu...
# history = model.fit([x_train,x_train,x_train,x_train,x_train],y_train, 
#                       batch_size=512, 
#                       validation_split=0.2,
#                       epochs=100,
#                       #overfits rather soon
#                       callbacks=[EarlyStopping(patience=2)])

# print('training done....')
# _, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].plot(history.history['loss'], label='loss')
# ax[0].plot(history.history['val_loss'], label='val_loss')
# ax[0].legend()
# ax[0].set_title('loss')

# ax[1].plot(history.history['acc'], label='acc')
# ax[1].plot(history.history['val_acc'], label='val_acc')
# ax[1].legend()
# ax[1].set_title('acc')

# plt.show()
# #for train set
# y_pred = model.predict([x_train,x_train,x_train,x_train,x_train],batch_size=1024, verbose=1)
# search_result = threshold_search(y_train, y_pred)
# print(search_result)
# y_pred = y_pred>search_result['threshold']
# y_pred = y_pred.astype(int)

# print('RESULTS ON TRAINING SET:\n',classification_report(y_train,y_pred))


# #for test set
# y_pred = model.predict([x_test,x_test,x_test,x_test,x_test],batch_size=1024, verbose=1)
# search_result = threshold_search(y_test, y_pred)
# print(search_result)
# y_pred = y_pred>search_result['threshold']
# y_pred = y_pred.astype(int)

# print('RESULTS ON TEST SET:\n',classification_report(y_test,y_pred))
#fit final model on all data
print('text to sequence')
x = tokenizer.texts_to_sequences(df['question_text'])

print('pad sequence')
## Pad the sentences 
x = pad_sequences(x,maxlen=max_len)

## Get the target values
y = df['target'].values

print('fiting final model...')
history = model.fit([x,x,x,x,x],y, batch_size=512, epochs=4)

print('fitting on full data done...')
_, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(history.history['loss'], label='loss')
ax[0].legend()
ax[0].set_title('loss')

ax[1].plot(history.history['acc'], label='acc')
ax[1].legend()
ax[1].set_title('acc')

plt.show()

y_pred = model.predict([x,x,x,x,x],batch_size=1024, verbose=1)
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

y_pred = model.predict([x_final,x_final,x_final,x_final,x_final],batch_size=1024,verbose=1)
y_pred = y_pred > search_result['threshold']
y_pred = y_pred.astype(int)
print(y_pred[:5])

df_subm = pd.DataFrame()
df_subm['qid'] = df_final.qid
df_subm['prediction'] = y_pred
print(df_subm.head())
df_subm.to_csv('submission.csv', index=False)
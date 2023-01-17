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
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

print('loading word2vec model...')
word2vec = gensim.models.KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
print('vocab:',len(word2vec.vocab))
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
max_len = 200

print('Fiting tokenizer')
## Tokenize the sentences
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(df['question_text'])

print('spliting data')
df_train,df_test = train_test_split(df)

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

all_embs = word2vec.vectors
emb_mean,emb_std = all_embs.mean(), all_embs.std()
print(emb_mean,emb_std)

print(num_words,' from ',len(tokenizer.word_index.items()))
num_words = min(num_words, len(tokenizer.word_index))
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

from keras.layers import Dense, Input,Embedding, Dropout, Activation, CuDNNLSTM,BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D,GlobalAveragePooling1D
from keras.models import Model
from keras.callbacks import Callback,EarlyStopping
from keras.engine import Layer
from keras.initializers import Ones, Zeros
import keras.backend as K
from keras import regularizers
from keras import constraints

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

inp = Input(shape=(max_len,))
#classic emb layer with pretrained weights
x = Embedding(num_words, dim, weights=[embedding_matrix], trainable=True)(inp)
# x = BatchNormalization()(x)
x = GlobalAveragePooling1D()(x)
# x = BatchNormalization()(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train,y_train, 
                      batch_size=512, 
                      validation_split=0.2,
                      class_weight=class_weights,
                      epochs=1000,                     
                      callbacks=[EarlyStopping(patience=5)])

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

print('RESULTS ON TRAINING SET:',classification_report(y_train,y_pred))


#for test set
y_pred = model.predict(x_test,batch_size=1024, verbose=1)
search_result = threshold_search(y_test, y_pred)
print(search_result)
y_pred = y_pred>search_result['threshold']
y_pred = y_pred.astype(int)

print('RESULTS ON TEST SET:',classification_report(y_test,y_pred))

#fit final model on all data
print('text to sequence')
x = tokenizer.texts_to_sequences(df['question_text'])

print('pad sequence')
## Pad the sentences 
x = pad_sequences(x,maxlen=max_len)

## Get the target values
y = df['target'].values

print('fiting final model...')
model.fit(x,y, batch_size=512, epochs=13,class_weight=class_weights)

y_pred = model.predict(x,batch_size=1024, verbose=1)
y_pred = y_pred>search_result['threshold']
y_pred = y_pred.astype(int)

print(classification_report(y,y_pred))

#submission
print('Loading test data...')
df_final = pd.read_csv('../input/test.csv')
df_final["question_text"].fillna("_##_", inplace=True)

x_final=tokenizer.texts_to_sequences(df_final['question_text'])
x_final = pad_sequences(x_final,maxlen=max_len)

y_pred = model.predict(x_final)
y_pred = y_pred > search_result['threshold']
y_pred = y_pred.astype(int)
print(y_pred[:5])

df_subm = pd.DataFrame()
df_subm['qid'] = df_final.qid
df_subm['prediction'] = y_pred
print(df_subm.head())
df_subm.to_csv('submission.csv', index=False)
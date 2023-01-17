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

#inspired by https://www.tensorflow.org/tutorials/keras/basic_regression
class PrintDot(Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 80 == 0: 
        print('')
    print('.', end='')

#inspiration from https://stackoverflow.com/questions/29760935/how-to-get-vector-for-a-sentence-from-the-word2vec-of-tokens-in-sentence
def load_x_from_df(df,model,max_len):
    sequences = []
    for question_text in df['question_text'].values:
        tokens = simple_preprocess(question_text)
        sentence = []
        for word in tokens:
            # print(model.wv[word])
            if word in model.wv.vocab:
                sentence.append(model.wv[word])
        if len(sentence) == 0:
            sentence = np.zeros((max_len,300))
        sequences.append(np.mean(sentence,axis=1))
    
    return pad_sequences(sequences,dtype='float32',maxlen=max_len)
print('loading word2vec model...')
model = gensim.models.KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
print('vocab:',len(model.wv.vocab))
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

print('loading sequences...')
print('creating %d sequences'%len(df))
x = load_x_from_df(df,model,max_len)
print(x.shape)
y = df.target.values
print(y.shape)
indexes_to_remove = []
for i in range(len(x)):
    if np.sum(x[i]) == 0.:
        indexes_to_remove.append(i)

print(indexes_to_remove)

#when sequence contains only 0 masking would mask it and actually no input would be present for NN. So we need to remove those...
if len(indexes_to_remove) > 0:
    x = np.delete(x,indexes_to_remove,axis=0)
    print(x.shape)
    y = np.delete(y,indexes_to_remove)
    print(y.shape)
    
x_train,x_test,y_train,y_test = train_test_split(x,y)

print(np.unique(y_train,return_counts=True))
print(np.unique(y_test,return_counts=True))

print('Computing class weights....')
#https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_test),
                                                 y_test)
print('class_weights:',class_weights)
print('Creating model...')
#inpiration from : https://github.com/keras-team/keras/blob/master/examples/imdb_fasttext.py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
#4x256
# {'threshold': 0.11, 'f1': 0.20848428957249193}
#            1       0.16      0.31      0.21     20264
#4x512
# {'threshold': 0.13, 'f1': 0.20549967703238906}
#               precision    recall  f1-score   support

#            1       0.15      0.33      0.21     20264

#back to relu
#256 relu+batchnorm
# {'threshold': 0.1, 'f1': 0.21432501711110813}
#               precision    recall  f1-score   support
#             1       0.15      0.40      0.21     20195
# 32
# {'threshold': 0.09, 'f1': 0.2132783413837488}
#               precision    recall  f1-score   support
#            1       0.15      0.38      0.21     20195

#64
# {'threshold': 0.09, 'f1': 0.21638974447963216}
#               precision    recall  f1-score   support
#            1       0.15      0.37      0.22     20195

from keras.layers import BatchNormalization,Dropout,AlphaDropout
from keras.engine import Layer
from keras.initializers import Ones, Zeros

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


dnn_model = Sequential()
dnn_model.add(BatchNormalization(input_shape=(x.shape[1],)))
dnn_model.add(Dense(32, activation='relu'))
dnn_model.add(LayerNormalization())
dnn_model.add(Dense(1, activation='sigmoid'))

dnn_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(dnn_model.summary())
print('fiting model...')
history = dnn_model.fit(x_train,y_train,
                        validation_split=0.2,
                        class_weight=class_weights,
                        epochs=1000, 
                        callbacks=[EarlyStopping(patience=5)],
                        verbose=2,
                        batch_size = 64)

print('model score:',dnn_model.evaluate(x_test,y_test,verbose=0))

y_pred = dnn_model.predict(x_test)
search_result = threshold_search(y_test, y_pred)
print(search_result)
y_pred = y_pred>search_result['threshold']
y_pred = y_pred.astype(int)

print(classification_report(y_test,y_pred))

print(history.history.keys())

_,ax = plt.subplots(1,2,figsize=(12,6))
ax[0].plot(history.history['loss'],label='loss')
ax[0].plot(history.history['val_loss'],label='val_loss')
ax[0].legend()
ax[0].set_title('loss')
ax[1].plot(history.history['acc'],label='acc')
ax[1].plot(history.history['val_acc'],label='val_acc')
ax[1].legend()
ax[1].set_title('acc')
plt.show()

dnn_model.fit(x,y,class_weight=class_weights,epochs=50, verbose=2,batch_size = 64)

print('Loading test data...')
df_test = pd.read_csv('../input/test.csv')
df_test["question_text"].fillna("_##_",inplace=True)

print('creating %d sequences'%len(df_test))
x_test = load_x_from_df(df_test,model,max_len)
print(x_test.shape)

indexes_to_remove = []
for i in range(len(x_test)):
    if np.sum(x_test[i]) == 0.:
        indexes_to_remove.append(i)

print(indexes_to_remove)

#when sequence contains only 0 masking would mask it and actually no input would be present for NN. So we need to remove those...
if len(indexes_to_remove) > 0:
    x_test = np.delete(x_test,indexes_to_remove,axis=0)
    print(x_test.shape)

y_pred = dnn_model.predict(x_test)
print(y_pred[:5] > search_result['threshold'])
y_pred = y_pred > search_result['threshold']
y_pred = y_pred.astype(int)

df_subm = pd.DataFrame()
df_subm['qid'] = df_test.qid
df_subm['prediction']=y_pred
print(df_subm.head())
df_subm.to_csv('submission.csv',index=False)
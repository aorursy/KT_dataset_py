import numpy as np
import pandas as pd
import gc
import re
import operator 
import string
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
from keras.engine.topology import Layer
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import VotingClassifier, BaggingClassifier
import xgboost as xgb
tqdm.pandas()
embedding_size = 300 # how big is each word vector
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
max_len = 100 # max number of words in a question to use
def clean_text(x):
    x = str(x)
    for p in string.punctuation + '“…':
        x = x.replace(p, ' ' + p + ' ')
    
    x = x.replace('_', '')
    
    x = re.sub("`","'", x)
    x = re.sub("(?i)n\'t",' not', x)
    x = re.sub("(?i)\'re",' are', x)
    x = re.sub("(?i)\'s",' is', x)
    x = re.sub("(?i)\'d",' would', x)
    x = re.sub("(?i)\'ll",' will', x)
    x = re.sub("(?i)\'t",' not', x)
    x = re.sub("(?i)\'ve",' have', x)
    x = re.sub("(?i)\'m",' am', x)
    
    x = re.sub("(?i)n\’t",' not', x)
    x = re.sub("(?i)\’re",' are', x)
    x = re.sub("(?i)\’s",' is', x)
    x = re.sub("(?i)\’d",' would', x)
    x = re.sub("(?i)\’ll",' will', x)
    x = re.sub("(?i)\’t",' not', x)
    x = re.sub("(?i)\’ve",' have', x)
    x = re.sub("(?i)\’m",' am', x)
    
    x = re.sub('(?i)Quorans','Quora', x)
    x = re.sub('(?i)Qoura','Quora', x)
    x = re.sub('(?i)Quoran','Quora', x)
    x = re.sub('(?i)dropshipping','drop shipping', x)
    x = re.sub('(?i)HackerRank','Hacker Rank', x)
    x = re.sub('(?i)Unacademy','un academy', x)
    x = re.sub('(?i)eLitmus','India hire employees', x)
    x = re.sub('(?i)WooCommerce','Commerce', x)
    x = re.sub('(?i)hairfall','hair fall', x)
    x = re.sub('(?i)marksheet','mark sheet', x)
    x = re.sub('(?i)articleship','article ship', x)
    x = re.sub('(?i)cryptocurrencies','cryptocurrency', x)
    x = re.sub('(?i)coinbase','cryptocurrency', x)
    x = re.sub('(?i)altcoin','bitcoin', x)
    x = re.sub('(?i)altcoins','bitcoins', x)
    x = re.sub('(?i)litecoin','bitcoin', x)
    x = re.sub('(?i)litecoins','bitcoins', x)
    x = re.sub('(?i)demonetisation','demonetization', x)
    x = re.sub('(?i)ethereum','bitcoin', x)
    x = re.sub('(?i)ethereums','bitcoins', x)
    x = re.sub('(?i)quorans','quora', x)
    x = re.sub('(?i)Brexit','britan exit', x)
    x = re.sub('(?i)upwork','freelance', x)
    x = re.sub('(?i)Unacademy','un academy', x)
    x = re.sub('(?i)Blockchain','blockchain', x)
    x = re.sub('(?i)GDPR','General Data Protection Regulation', x)
    x = re.sub('(?i)Qoura','quora', x)
    x = re.sub('(?i)HackerRank','Hacker Rank', x)
    x = re.sub('(?i)Cryptocurrency','cryptocurrency', x)
    x = re.sub('(?i)Binance','cryptocurrency', x)
    x = re.sub('(?i)Redmi','mobile phone', x)
    x = re.sub('(?i)TensorFlow','Tensor Flow', x)
    x = re.sub('(?i)Golang','programming language', x)
    x = re.sub('(?i)eLitmus','India hire employees', x)
    
    return x
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')
# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
def get_model_1(embedding_matrix):
    inp = Input(shape=(max_len, ))
    x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(80, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    
    y = Conv1D(64, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x)
    y = Dense(16, activation='relu')(y)
    y = Flatten()(y)
    
    atten = Attention(max_len)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    conc = concatenate([atten, avg_pool, max_pool])
#     conc = Conv2D(64, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(conc)
    conc = Dense(64, activation='relu')(conc)
    
    conc_z = concatenate([conc, y])
    conc_z = Dense(32, activation='relu')(conc_z)
    conc_z = Dropout(0.1)(conc_z)
    output = Dense(1, activation='sigmoid')(conc_z)
    
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
# https://www.kaggle.com/shujian/fork-of-mix-of-nn-models

def get_model_2(embedding_matrix):
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(max_len)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
# https://www.kaggle.com/yekenot/2dcnn-textclassifier
    
def get_model_3(embedding_matrix):
    filter_sizes = [1,2,3,5]
    num_filters = 42

    inp = Input(shape=(max_len, ))
    x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Reshape((max_len, embedding_size, 1))(x)
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_size),
                                 kernel_initializer='he_normal', activation='tanh')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_size),
                                 kernel_initializer='he_normal', activation='tanh')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_size), 
                                 kernel_initializer='he_normal', activation='tanh')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embedding_size),
                                 kernel_initializer='he_normal', activation='tanh')(x)
    
    maxpool_0 = MaxPool2D(pool_size=(max_len - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(max_len - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_len - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(max_len - filter_sizes[3] + 1, 1))(conv_3)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = Dropout(0.1)(z) 
    output = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model
# https://www.kaggle.com/shujian/single-rnn-model-with-meta-features?scriptVersionId=7593124

def get_model_4(embedding_matrix):
    inp = Input(shape=(max_len, ))
    x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
    
    atten = Attention(max_len)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    conc = concatenate([atten, avg_pool, max_pool])
    conc = Dense(64, activation='relu')(conc)
    conc = Dropout(0.1)(conc)
    output = Dense(1, activation='sigmoid')(conc)
    
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# https://www.kaggle.com/suicaokhoailang/beating-the-baseline-with-one-weird-trick-0-691

def get_model_5(embedding_matrix):
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(max_len)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.1)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
def get_model_6(embedding_matrix):
    inp = Input(shape=(max_len, ))
    x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.15)(x)
    x = Bidirectional(CuDNNLSTM(max_len, return_sequences=True))(x)
    x = Conv1D(64, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    output = Dense(1, activation='sigmoid')(conc)
    
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def get_f1_metric(y_true, y_predicted):        
    f1_metrics = []
    for thresh in np.arange(0.1, 0.51, 0.01):
        thresh = np.round(thresh, 2)
        f1_metric = f1_score(y_true, (y_predicted>thresh).astype(int))
        f1_metrics.append((thresh, f1_metric))
        print('F1 score at threshold {0} is {1}'.format(thresh, f1_metric))
    
    threshold = max(f1_metrics, key=lambda x: x[1])[0]
    
    print(threshold)
    
    return threshold
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

print('Train shape : ', train_df.shape)
print('Test shape : ', test_df.shape)
train_df['question_text'] = train_df['question_text'].progress_apply(lambda x: clean_text(x))
test_df['question_text'] = test_df['question_text'].progress_apply(lambda x: clean_text(x))
train_df, val_df = train_test_split(train_df, 
                                    test_size=0.07, 
                                    random_state=42, 
                                    stratify=train_df['target'])

X_train = train_df['question_text'].values
y_train = train_df['target'].values
X_val = val_df['question_text'].values
y_val = val_df['target'].values
X_test = test_df['question_text'].values

print(X_train.shape, 
      y_train.shape, 
      X_val.shape, 
      y_val.shape, 
      X_test.shape)
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_val))
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=max_len)
X_val = pad_sequences(X_val, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

word_index = tokenizer.word_index

print(X_train.shape, X_val.shape, X_test.shape, len(word_index))
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

nb_words = min(max_features, len(word_index))
embedding_matrix_glove = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in tqdm(word_index.items()):
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_glove[i] = embedding_vector

del embeddings_index
gc.collect() 

print(embedding_matrix_glove.shape)
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

nb_words = min(max_features, len(word_index))
embedding_matrix_wiki = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in tqdm(word_index.items()):
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_wiki[i] = embedding_vector

del embeddings_index, word_index
gc.collect()   

print(embedding_matrix_wiki.shape)
embedding_matrix = np.mean([embedding_matrix_glove, embedding_matrix_wiki], axis = 0)
np.shape(embedding_matrix)
model_1 = get_model_1(embedding_matrix_glove)
model_1.fit(X_train, y_train, batch_size=512, epochs=3, validation_data=(X_val, y_val))
model_2 = get_model_2(embedding_matrix_glove)
model_2.fit(X_train, y_train, batch_size=512, epochs=2, validation_data=(X_val, y_val))
model_3 = get_model_3(embedding_matrix)
model_3.fit(X_train, y_train, batch_size=512, epochs=3, validation_data=(X_val, y_val))
model_4 = get_model_4(embedding_matrix)
model_4.fit(X_train, y_train, batch_size=512, epochs=2, validation_data=(X_val, y_val))
model_5 = get_model_5(embedding_matrix_wiki)
model_5.fit(X_train, y_train, batch_size=512, epochs=3, validation_data=(X_val, y_val))
model_6 = get_model_6(embedding_matrix_wiki)
model_6.fit(X_train, y_train, batch_size=512, epochs=2, validation_data=(X_val, y_val))
pred_val_1 = model_1.predict([X_val], batch_size=512, verbose=1)
pred_val_2 = model_2.predict([X_val], batch_size=512, verbose=1)
pred_val_3 = model_3.predict([X_val], batch_size=512, verbose=1)
pred_val_4 = model_4.predict([X_val], batch_size=512, verbose=1)
pred_val_5 = model_5.predict([X_val], batch_size=512, verbose=1)
pred_val_6 = model_6.predict([X_val], batch_size=512, verbose=1)
pred_val_1_df = np.reshape(pred_val_1, (pred_val_1.shape[0]))
pred_val_2_df = np.reshape(pred_val_2, (pred_val_2.shape[0]))
pred_val_3_df = np.reshape(pred_val_3, (pred_val_3.shape[0]))
pred_val_4_df = np.reshape(pred_val_4, (pred_val_4.shape[0]))
pred_val_5_df = np.reshape(pred_val_5, (pred_val_5.shape[0]))
pred_val_6_df = np.reshape(pred_val_6, (pred_val_6.shape[0]))
pred_val_df = np.reshape(y_val, (y_val.shape[0]))

validation_df = pd.DataFrame({'val_1': pred_val_1_df, 'val_2': pred_val_2_df, 'val_3': pred_val_3_df, 'val_4': pred_val_4_df, 'val_5': pred_val_5_df, 'val_6': pred_val_6_df, 'prediction': pred_val_df})
validation_df.to_csv('validation.csv', index=False)
validation_df['val_12_mean'] = (validation_df['val_1'] + validation_df['val_2']) / 2.0
validation_df['val_34_mean'] = (validation_df['val_3'] + validation_df['val_4']) / 2.0
validation_df['val_56_mean'] = (validation_df['val_5'] + validation_df['val_6']) / 2.0
validation_df['val_123_mean'] = (validation_df['val_1'] + validation_df['val_2'] + validation_df['val_3']) / 3.0
validation_df['val_456_mean'] = (validation_df['val_4'] + validation_df['val_5'] + validation_df['val_6']) / 3.0
validation_df['val_1_log'] = np.log(validation_df['val_1'])
validation_df['val_2_log'] = np.log(validation_df['val_2'])
validation_df['val_3_log'] = np.log(validation_df['val_3'])
validation_df['val_4_log'] = np.log(validation_df['val_4'])
validation_df['val_5_log'] = np.log(validation_df['val_5'])
validation_df['val_6_log'] = np.log(validation_df['val_6'])
validation_df['val_12_log_mean'] = (validation_df['val_1_log'] + validation_df['val_2_log']) / 2.0
validation_df['val_34_log_mean'] = (validation_df['val_3_log'] + validation_df['val_4_log']) / 2.0
validation_df['val_56_log_mean'] = (validation_df['val_5_log'] + validation_df['val_6_log']) / 2.0
validation_df['val_123_log_mean'] = (validation_df['val_1_log'] + validation_df['val_2_log'] + validation_df['val_3_log']) / 3.0
validation_df['val_456_log_mean'] = (validation_df['val_4_log'] + validation_df['val_5_log'] + validation_df['val_6_log']) / 3.0

validation_df = validation_df[[c for c in validation_df.columns if c != 'prediction'] + ['prediction']]

validation_df.head()
X_val = validation_df[validation_df.columns.values[:-1]].values
y_val = validation_df[validation_df.columns.values[-1]].values

clf = VotingClassifier(estimators=
                       [('gnb', GaussianNB()), 
                        ('xgb', xgb.XGBClassifier(max_dept=100, n_estimators=15, learning_rate=0.05, 
                                                  colsample_bytree=0.5, gamma=0.01, reg_alpha=4, 
                                                  objective='binary:logistic')), 
                        ('knn', KNeighborsClassifier(n_neighbors=500)), 
                        ('rf', RandomForestClassifier(max_depth=3, n_estimators=100)), 
                        ('lr', LogisticRegression()), 
                        ('dt', DecisionTreeClassifier(max_depth=3, criterion='entropy')),
                        ('adb', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3, criterion='entropy'), learning_rate=0.05)),
                        ('gb', GradientBoostingClassifier(max_depth=3, learning_rate=0.05, 
                                                          n_estimators=100)),
                        ('qda', QuadraticDiscriminantAnalysis()),
                        ('lda', LinearDiscriminantAnalysis())
                       ],
                       voting='soft')
clf.fit(X_val, y_val)
pred_test_1 = model_1.predict([X_test], batch_size=512, verbose=1)
pred_test_2 = model_2.predict([X_test], batch_size=512, verbose=1)
pred_test_3 = model_3.predict([X_test], batch_size=512, verbose=1)
pred_test_4 = model_4.predict([X_test], batch_size=512, verbose=1)
pred_test_5 = model_5.predict([X_test], batch_size=512, verbose=1)
pred_test_6 = model_6.predict([X_test], batch_size=512, verbose=1)
pred_test_1_df = np.reshape(pred_test_1, (pred_test_1.shape[0]))
pred_test_2_df = np.reshape(pred_test_2, (pred_test_2.shape[0]))
pred_test_3_df = np.reshape(pred_test_3, (pred_test_3.shape[0]))
pred_test_4_df = np.reshape(pred_test_4, (pred_test_4.shape[0]))
pred_test_5_df = np.reshape(pred_test_5, (pred_test_5.shape[0]))
pred_test_6_df = np.reshape(pred_test_6, (pred_test_6.shape[0]))

test_df1 = pd.DataFrame({'val_1': pred_test_1_df, 'val_2': pred_test_2_df, 'val_3': pred_test_3_df, 'val_4': pred_test_4_df, 'val_5': pred_test_5_df, 'val_6': pred_test_6_df})
test_df1.to_csv('test.csv', index=False)
test_df1['val_12_mean'] = (test_df1['val_1'] + test_df1['val_2']) / 2.0
test_df1['val_34_mean'] = (test_df1['val_3'] + test_df1['val_4']) / 2.0
test_df1['val_56_mean'] = (test_df1['val_5'] + test_df1['val_6']) / 2.0
test_df1['val_123_mean'] = (test_df1['val_1'] + test_df1['val_2'] + test_df1['val_3']) / 3.0
test_df1['val_456_mean'] = (test_df1['val_4'] + test_df1['val_5'] + test_df1['val_6']) / 3.0
test_df1['val_1_log'] = np.log(test_df1['val_1'])
test_df1['val_2_log'] = np.log(test_df1['val_2'])
test_df1['val_3_log'] = np.log(test_df1['val_3'])
test_df1['val_4_log'] = np.log(test_df1['val_4'])
test_df1['val_5_log'] = np.log(test_df1['val_5'])
test_df1['val_6_log'] = np.log(test_df1['val_6'])
test_df1['val_12_log_mean'] = (test_df1['val_1_log'] + test_df1['val_2_log']) / 2.0
test_df1['val_34_log_mean'] = (test_df1['val_3_log'] + test_df1['val_4_log']) / 2.0
test_df1['val_56_log_mean'] = (test_df1['val_5_log'] + test_df1['val_6_log']) / 2.0
test_df1['val_123_log_mean'] = (test_df1['val_1_log'] + test_df1['val_2_log'] + test_df1['val_3_log']) / 3.0
test_df1['val_456_log_mean'] = (test_df1['val_4_log'] + test_df1['val_5_log'] + test_df1['val_6_log']) / 3.0

test_df1.head()
X_test = test_df1.values

y_test_pred = clf.predict(X_test)
y_test_pred = np.reshape(y_test_pred, (y_test_pred.shape[0]))
submission_df = pd.DataFrame({'qid': test_df['qid'].values, 'prediction': y_test_pred})
submission_df.to_csv('submission.csv', index=False)
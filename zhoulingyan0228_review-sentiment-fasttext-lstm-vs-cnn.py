import numpy as n
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import keras
import keras.backend as K
from keras.layers import *
from keras.losses import *
from keras.models import *
from keras.callbacks import *
from keras.activations import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep='\t')
test = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep='\t')

train.head()
def get_preprocessing_func():
    tokenizer = WordPunctTokenizer()
    lemmatizer = WordNetLemmatizer()
    def preprocessing_func(sent):
        return [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(sent)]
    return preprocessing_func

X = train['Phrase'].apply(get_preprocessing_func()).values
y = train['Sentiment'].values
X_test = test['Phrase'].apply(get_preprocessing_func()).values
def prepare_tokenizer_and_weights(X):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(X)
    
    weights = np.zeros((len(tokenizer.word_index)+1, 300))
    with open("../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec") as f:
        next(f)
        for l in f:
            w = l.split(' ')
            if w[0] in tokenizer.word_index:
                weights[tokenizer.word_index[w[0]]] = np.array([float(x) for x in w[1:301]])
    return tokenizer, weights
tokenizer, weights = prepare_tokenizer_and_weights(np.append(X, X_test))
X_seq = tokenizer.texts_to_sequences(X)
MAX_LEN = max(map(lambda x: len(x), X_seq))
X_seq = pad_sequences(X_seq, MAX_LEN)
MAX_ID = len(tokenizer.word_index)
print('MAX_LEN=', MAX_LEN)
print('MAX_ID=', MAX_ID)
def make_fast_text():
    fast_text = Sequential()
    fast_text.add(InputLayer((MAX_LEN,))) 
    fast_text.add(Embedding(input_dim=MAX_ID+1, output_dim=300, weights=[weights], trainable=True))
    fast_text.add(SpatialDropout1D(0.5))
    fast_text.add(GlobalMaxPooling1D())
    fast_text.add(Dropout(0.5))
    fast_text.add(Dense(5,activation='softmax'))
    return fast_text

fast_texts = [make_fast_text() for i in range(3)]
fast_texts[0].summary()

for fast_text in fast_texts:
    X_seq_train, X_seq_valid, y_train, y_valid = train_test_split(X_seq, y, test_size=0.1)
    fast_text.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    fast_text.fit(X_seq_train, y_train, validation_data=(X_seq_valid, y_valid),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0)],
                 epochs=30, 
                 verbose=2)
def make_model_lstm():
    model_lstm = Sequential()
    model_lstm.add(InputLayer((MAX_LEN,))) 
    model_lstm.add(Embedding(input_dim=MAX_ID+1, output_dim=300, weights=[weights], trainable=True))
    model_lstm.add(SpatialDropout1D(0.5))
    model_lstm.add(Bidirectional(CuDNNLSTM(300, return_sequences=True)))
    model_lstm.add(BatchNormalization())
    model_lstm.add(SpatialDropout1D(0.5))
    model_lstm.add(Bidirectional(CuDNNLSTM(300)))
    model_lstm.add(BatchNormalization())
    model_lstm.add(Dropout(0.5))
    model_lstm.add(Dense(5,activation='softmax'))
    return model_lstm

model_lstms = [make_model_lstm() for i in range(2)]
model_lstms[0].summary()

for model_lstm in model_lstms:
    X_seq_train, X_seq_valid, y_train, y_valid = train_test_split(X_seq, y, test_size=0.1)
    model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model_lstm.fit(X_seq_train, y_train, validation_data=(X_seq_valid, y_valid),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=1, verbose=0)],
                 epochs=30, 
                 verbose=2)
def make_model_cnn():
    inputs = Input((MAX_LEN,))
    x = Embedding(input_dim=MAX_ID+1, output_dim=300, weights=[weights], trainable=True)(inputs)
    x = SpatialDropout1D(0.5)(x)
    x = Conv1D(300, kernel_size=5,activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(0.5)(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Conv1D(300, kernel_size=5,activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(5,activation='softmax')(x)
    model_cnn = Model(inputs, outputs)
    return model_cnn

model_cnns = [make_model_cnn() for i in range(3)]
model_cnns[0].summary()

for model_cnn in model_cnns:
    X_seq_train, X_seq_valid, y_train, y_valid = train_test_split(X_seq, y, test_size=0.1)
    model_cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model_cnn.fit(X_seq_train, y_train, validation_data=(X_seq_valid, y_valid),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0)],
                 epochs=30, 
                 verbose=2)
def make_model_abcnn():
    def attention_layer(l):
        x = Permute((2,1))(l)
        x = Dense(K.int_shape(x)[2], activation='sigmoid')(x)
        x = Permute((2,1))(x)
        return multiply([x, l])
    inputs = Input((MAX_LEN,))
    x = Embedding(input_dim=MAX_ID+1, output_dim=300, weights=[weights], trainable=True)(inputs)
    x = SpatialDropout1D(0.5)(x)
    x = Conv1D(300, kernel_size=3,activation='relu')(x)
    x = attention_layer(x)
    x = SpatialDropout1D(0.5)(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Conv1D(300, kernel_size=3,activation='relu')(x)
    x = attention_layer(x)
    x = SpatialDropout1D(0.5)(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Conv1D(300, kernel_size=3,activation='relu')(x)
    x = attention_layer(x)
    x = GlobalMaxPooling1D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(5,activation='softmax')(x)
    model_cnn = Model(inputs, outputs)
    return model_cnn

model_abcnns = [make_model_abcnn() for i in range(3)]
model_abcnns[0].summary()

for model_abcnn in model_abcnns:
    X_seq_train, X_seq_valid, y_train, y_valid = train_test_split(X_seq, y, test_size=0.1)
    model_abcnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model_abcnn.fit(X_seq_train, y_train, validation_data=(X_seq_valid, y_valid),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0)],
                 epochs=30, 
                 verbose=2)
def make_model_bagged(models):
    inputs = Input((MAX_LEN,))
    outputs = average([model(inputs) for model in models])
    return Model(inputs, outputs)
model_bagged = make_model_bagged(model_abcnns)
model_bagged.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
y_prob = model_bagged.predict(X_seq)
y_predict = np.argmax(y_prob, axis=1)
print(classification_report(y, y_predict))
sns.heatmap(confusion_matrix(y, y_predict));
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_seq = pad_sequences(X_test_seq, MAX_LEN)
y_test_prob = model_bagged.predict(X_test_seq)
y_test_predict = np.argmax(y_test_prob, axis=1)
out_df = test[['PhraseId']]
out_df['Sentiment'] = y_test_predict
out_df.to_csv('submission.csv', index=False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# NLTK modules
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer, WordNetLemmatizer

import re

from gensim.models import Word2Vec # Word2Vec module
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, remove_stopwords, strip_numeric, stem_text
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import text, sequence
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Training data
train_df = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/train.csv')
test_df = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv')

submission_df = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/sample_submission_UVKGLZE.csv')
print(train_df.isnull().sum())
print(train_df.columns)
# Converting binary column to category
target_cols = ['Computer Science', 'Physics', 'Mathematics','Statistics', 'Quantitative Biology', 'Quantitative Finance']

y_data = train_df[target_cols]
# Plot category data
plt.figure(figsize=(10,6))
y_data.sum(axis=0).plot.bar()
plt.show()

# Stemmer object
porter = PorterStemmer()
wnl = WordNetLemmatizer()

class DataPreprocess:
    
    def __init__(self):
        self.filters = [strip_tags,
                       strip_numeric,
                       strip_punctuation,
                       lambda x: x.lower(),
                       lambda x: re.sub(r'\s+\w{1}\s+', '', x),
                       remove_stopwords]
    def __call__(self, doc):
        clean_words = self.__apply_filter(doc)
        return clean_words
    
    def __apply_filter(self, doc):
        try:
            cleanse_words = set(preprocess_string(doc, self.filters))
#             filtered_words = set(wnl.lemmatize(w) if w.endswith('e') or w.endswith('y') else porter.stem(w) for w in cleanse_words)
            return ' '.join(cleanse_words)
        except TypeError as te:
            raise(TypeError("Not a valid data {}".format(te)))
train_df['train_or_test'] = 0
test_df['train_or_test'] = 1

feature_col = ['ID', 'TITLE', 'ABSTRACT', 'train_or_test']
# Concat train and test data
combined_set = pd.concat([train_df[feature_col], test_df[feature_col]])
combined_set
# Combine the Title and Abstract data
combined_set['TEXT'] = combined_set['TITLE'] + combined_set['ABSTRACT']

# articles['Processed'] = articles['TEXT'].apply(DataPreprocess())
# Drop unwanted columns
combined_set = combined_set.drop(['TITLE', 'ABSTRACT'], axis=1)

# Invoke data preprocess operation on the text data
combined_set['Processed'] = combined_set['TEXT'].apply(DataPreprocess())
combined_set.columns
train_set = combined_set.loc[combined_set['train_or_test'] == 0]
test_set = combined_set.loc[combined_set['train_or_test'] == 1]
# Drop key reference column
train_set = train_set.drop('train_or_test', axis=1)
test_set = test_set.drop('train_or_test', axis=1)
train_set[0:2].values
train_data = train_set['Processed']
test_data = test_set['Processed']

y = y_data.values

X_train, X_valid, y_train, y_valid = train_test_split(train_data, y, test_size=0.3, random_state=42)

def label_encoding(y_train):
    """
        Encode the given list of class labels
        :y_train_enc: returns list of encoded classes
        :labels: actual class labels
    """
    lbl_enc = LabelEncoder()
    
    y_train_enc = lbl_enc.fit_transform(y_train)
    labels = lbl_enc.classes_
    
    return y_train_enc, labels


def word_embedding(train, test, max_features, max_len=200):
    try:
        # Keras Tokenizer class object
        tokenizer = text.Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(train)
        
        train_data = tokenizer.texts_to_sequences(train)
        test_data = tokenizer.texts_to_sequences(test)
        
        # Get the max_len
        vocab_size = len(tokenizer.word_index) + 1
        
        # Padd the sequence based on the max-length
        x_train = sequence.pad_sequences(train_data, maxlen=max_len, padding='post')
        x_test = sequence.pad_sequences(test_data, maxlen=max_len, padding='post')
        # Return train, test and vocab size
        return tokenizer, x_train, x_test, vocab_size
    except ValueError as ve:
        raise(ValueError("Error in word embedding {}".format(ve)))
max_features = 6000
max_len = 200

tokenizer, x_pad_train, x_pad_valid, vocab_size = word_embedding(X_train, X_valid, max_features)
x_pad_train.shape
print("Vocab size: {}".format(vocab_size))
# def build_rnn(vocab_size,output_dim, max_len):
#     # Building RNN model
#     model = Sequential([
#         keras.layers.Embedding(vocab_size,200,
#                               input_length=max_len),
#         keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True)),
#         keras.layers.GlobalMaxPool1D(), # Remove flatten layer
#         keras.layers.Dense(128, activation='relu'),
#         keras.layers.Dropout(0.4),
#         keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)),
#         keras.layers.Dense(output_dim, activation='sigmoid')
#     ])

#     return model

def build_rnn(vocab_size,output_dim, max_len):
    # Building RNN model
    model = Sequential([
        keras.layers.Embedding(vocab_size,200,
                              input_length=max_len),
        keras.layers.BatchNormalization(),
        keras.layers.Bidirectional(keras.layers.LSTM(256,return_sequences=True)),
        keras.layers.GlobalMaxPool1D(), # Remove flatten layer
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(output_dim, activation='sigmoid')
    ])

    return model
rnn_model = build_rnn(vocab_size, 6, max_len)

# Summary of the model
rnn_model.summary()
# Compile the model
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = rnn_model.fit(x_pad_train, 
                        y_train,
                        batch_size=256,
                       epochs=7,
                       verbose=1,
                       validation_split=0.2)
score = rnn_model.evaluate(x_pad_valid, y_valid, verbose=1)

print("Loss:%.3f Accuracy: %.3f" % (score[0], score[1]))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
len(test_data)
# tokenizer = text.Tokenizer(num_words=5000)
tokenizer.fit_on_sequences(test_data)

X_test = tokenizer.texts_to_sequences(test_data)
x_pad_test = sequence.pad_sequences(X_test, maxlen=max_len, padding='post')
x_pad_test
y_preds = rnn_model.predict(x_pad_test)
for arr in y_preds:
    for i in range(len(arr)):
        if arr[i]>0.5:
            arr[i] = 1
        else:
            arr[i] = 0
y_preds = y_preds.astype('int32')
y_preds
pred_df = pd.DataFrame(y_preds, columns=target_cols)
submission_df[target_cols] = pred_df[target_cols]
submission_df
submission_df.to_csv("rnn_model_04.csv", index=False)

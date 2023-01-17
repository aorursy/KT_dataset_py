import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from nltk.stem import WordNetLemmatizer

# Tensorflow libraries
# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import text, sequence
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers

import tensorflow_hub as hub


# sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score

from gensim.models import Word2Vec # Word2Vec module
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, remove_stopwords, strip_numeric, stem_text


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/fake-news-content-detection/train.csv")
test_data = pd.read_csv("/kaggle/input/fake-news-content-detection/test.csv")
submission_data = pd.read_csv("/kaggle/input/fake-news-content-detection/sample submission.csv")
# Sample data from training data
train_data.sample(3)

# Dataset information
train_data.info()
train_data[train_data.duplicated(['Text'])]
train_data = train_data.drop_duplicates(['Text'])
train_data.sample(3)
train_data['NewsText'] = train_data['Text_Tag'].astype(str) +" "+ train_data['Text']
test_data['NewsText'] = test_data['Text_Tag'].astype(str) +" "+ test_data['Text']
# Stemmer object
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
            filtered_words = set(wnl.lemmatize(w, 'v') for w in cleanse_words)
            return ' '.join(cleanse_words)
        except TypeError as te:
            raise(TypeError("Not a valid data {}".format(te)))
train_data['Processed'] = train_data['NewsText'].apply(DataPreprocess())
test_data['Processed'] = test_data['NewsText'].apply(DataPreprocess())

# train_data['Processed'] = train_data['Text'].apply(DataPreprocess())
# test_data['Processed'] = test_data['Text'].apply(DataPreprocess())
test_data['Processed']
X = train_data['Processed']
y = train_data['Labels']

y_category = keras.utils.to_categorical(y, 6)

# Split data into Train and Holdout as 80:20 ratio
X_train, X_valid, y_train, y_valid = train_test_split(X, y_category, shuffle=True, test_size=0.33, random_state=111)

print("Train shape : {}, Holdout shape: {}".format(X_train.shape, X_valid.shape))
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

max_features = 5000
max_len = 128
output_dim = len(np.unique(y))

# Test data
X_test = test_data['Processed']

tokenizer, x_pad_train, x_pad_valid, vocab_size = word_embedding(X_train, X_valid, max_features)
# Test data
X_test = test_data['Processed']

tokenizer.fit_on_sequences(X_test)

X_test_seq = tokenizer.texts_to_sequences(X_test)
x_pad_test = sequence.pad_sequences(X_test_seq, maxlen=max_len, padding='post')
def compute_classweights(target):
    """
    Computes the weights of the target values based on the samples
    :param target: Y-target variable
    :return: dictionary object
    """
    # compute class weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(target),
                                                     target)
    
    # make the class weight list into dictionary
    weights = {}
    
    # enumerate the list
    for index, weight in enumerate(class_weights):
        weights[index] = weight
        
    return weights

# Get the class weights for the target variable
weights = compute_classweights(y)
weights
X_train.sample(3)
def build_rnn(vocab_size, output_dim, max_len):
    # Building RNN model
    model = Sequential([
        keras.layers.Embedding(vocab_size,128,
                              input_length=max_len),
        keras.layers.BatchNormalization(),
#         keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True)),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.002)),
        keras.layers.GlobalMaxPool1D(), # Remove flatten layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.002)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.002)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(output_dim, activation='softmax')
    ])

    return model
rnn_model = build_rnn(vocab_size, output_dim, max_len)

# Summary of the model
rnn_model.summary()
# Compile the model
rnn_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), 
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True), 
                  metrics=[tf.metrics.AUC()])
history = rnn_model.fit(x_pad_train, 
                        y_train,
                        batch_size=512,
                        epochs=20,
                        verbose=1,
                        validation_data=(x_pad_valid, y_valid),
                       class_weight=weights)
results = rnn_model.evaluate(x_pad_valid, y_valid)
y_preds = rnn_model.predict_proba(x_pad_test, batch_size=256)
y_preds[:,0]
final_df = pd.DataFrame({'0': y_preds[:,0],
                        '1': y_preds[:,1],
                        '2': y_preds[:,2],
                        '3': y_preds[:,3],
                        '4': y_preds[:,4],
                        '5': y_preds[:,5]}, index=test_data.index)
final_df
final_df.to_csv("fake_news_ann_08.csv", index=False)
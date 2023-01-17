# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
########################################
# Load the packages
########################################
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns

from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Bidirectional, GRU, Conv1D, GlobalMaxPooling1D, Dropout, TimeDistributed
from keras.layers.merge import concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

%matplotlib inline
########################################
# Define the hyper parameters
########################################
path = '../input/jigsaw-toxic-comment-classification-challenge/'
TRAIN_DATA_FILE= os.path.join(path, 'train.csv')
TEST_DATA_FILE= os.path.join(path, 'test.csv')

MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 50
########################################
# Load the training / testing set with pandas csv format
########################################
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)
print("A quick view of training set")
train_df.head()
print("A quick view of testing set")
test_df.head()
# What would be toxic?
train_df[train_df.toxic == 1].head(10)
'''
What's the positive ratio of each class ?
'''
def get_pos_ratio(data):
    return data.sum() / len(data)

pos_ratio = []
for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    pos_ratio.append(get_pos_ratio(train_df[col]))
assert pos_ratio[0] == 0.09584448302009764, "The answer is not correct."
print("Congrats, you passed the test.")
x = train_df.iloc[:,2:].sum()

plt.figure(figsize=(8,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)

rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
corr=train_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True)
########################################
## Text pre-processing and cleaning
########################################
print('Processing text dataset')
from collections import defaultdict

# regex to remove all Non-Alpha Numeric and space
special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)

# regex to replace all numeric
replace_numbers=re.compile(r'\d+',re.IGNORECASE)

def clean_text(text, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"i’m", "i am", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = replace_numbers.sub('', text)
    text = special_character_removal.sub('',text)
    
    return text
'''
Apply preprocessing and extract the training sentences and testing senteces from pandas dataframe.
Note that there are some N/A comment in the train/test set. Fill them up first.
'''
train_comments = []
test_comments = []
list_sentences_train = train_df["comment_text"].fillna("no comment").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_labels = train_df[list_classes].values
list_sentences_test = test_df["comment_text"].fillna("no comment").values

train_comments = [clean_text(text) for text in list_sentences_train]
test_comments = [clean_text(text) for text in list_sentences_test]
assert len(train_comments) == 159571 and len(test_comments) == 153164, "It seems that you lost some data."
assert 'E' not in train_comments[0], "It seems you did not preprocess the sentecnes. I found a upper case alphabet in your train set."
for i in range(3):
    print("Cleaned\n", train_comments[i] + '\n')
    print("Raw\n", train_df.iloc[i]['comment_text'] + '\n')
    print("------------------")
# Create a tokenize, which transforms a sentence to a list of ids
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
# Build the relation between words and ids 
tokenizer.fit_on_texts(train_comments + test_comments)
print(tokenizer.word_index['the']) # map 'the' to 1, map 'to' to 2,......
print(tokenizer.word_index['to'])
# Transform training/testing sentences to training/testing sequences.
train_sequences = tokenizer.texts_to_sequences(train_comments)
test_sequences = tokenizer.texts_to_sequences(test_comments)
for i in range(1):
    print("Transformed\n", str(train_sequences[i]) + '\n')
    print("Cleaned\n", train_comments[i] + '\n')
    print("------------------")
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', train_data.shape)
print('Shape of label tensor:', train_labels.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of test_data tensor:', test_data.shape)
'''
Try to build a tokenzier, which transform [['Hello', 'World'], ['Greeting', 'my', 'friend'], ['Hello', 'have', 'a', 'nice', 'day']]
to a list of index sequences. Note that the index should start from 1 because 0 is reserverd for padding token for some framework.
'''
tests_input_sentences =  [['Hello', 'World'], ['Greeting', 'my', 'friend'], ['Hello', 'have', 'a', 'nice', 'day']]
transform_this_sentences = [['Hello', 'my', 'friend']]

def index_encoding(sentences, raw_sent):
    word2idx = {}
    idx2word = {}
    ctr = 1
    for sentence in sentences:
        for word in sentence:
            if word not in word2idx.keys():
                word2idx[word] = ctr
                idx2word[ctr] = word
                ctr += 1
    results = []
    for sent in raw_sent:
        results.append([word2idx[word] for word in sent])
    return results
transformed = index_encoding(tests_input_sentences, transform_this_sentences)
assert transformed == [[1, 4, 5]], "The answer is not correct."
print("Congrats, you passed the test.")
########################################
## Define the text rnn model structure
########################################
def get_text_rnn():
    recurrent_units = 48
    dense_units = 32
    output_units = 6
    
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(input_layer)
    
    x = Bidirectional(GRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Bidirectional(GRU(recurrent_units, return_sequences=False))(x)
    
    x = Dense(dense_units, activation="relu")(x)
    output_layer = Dense(output_units, activation="sigmoid")(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
########################################
## Define the text cnn model structure
########################################
def get_text_cnn():
    filter_nums = 120
    dense_units = 72
    output_units = 6
    
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,trainable=False,)(input_layer)
        
    conv_0 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")(embedding_layer)
    conv_1 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")(embedding_layer)
    conv_2 = Conv1D(filter_nums, 4, kernel_initializer="normal", padding="valid", activation="relu")(embedding_layer)

    maxpool_0 = GlobalMaxPooling1D()(conv_0)
    maxpool_1 = GlobalMaxPooling1D()(conv_1)
    maxpool_2 = GlobalMaxPooling1D()(conv_2)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2])
    h1 = Dense(units=dense_units, activation="relu")(merged_tensor)
    output = Dense(units=output_units, activation='sigmoid')(h1)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
########################################
## Define the text hybrid model structure
########################################
def get_hybrid_textnn():
    recurrent_units = 48
    dense_units = 32
    filter_nums = 64
    output_units = 6

    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(input_layer)
    
    x = Bidirectional(GRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")(x)    
    
    max_pool = GlobalMaxPooling1D()(x)
    max_pool = Dropout(0.5)(max_pool)
    
    output_layer = Dense(output_units, activation="sigmoid")(max_pool)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

########################################
## Construct the cross-validation framework
########################################
def _train_model_by_logloss(model, batch_size, train_x, train_y, val_x, val_y, fold_id):
    # set an early stopping checker.
    # the training phase would stop when validation log loss decreases continuously for `patience` rounds. 
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    bst_model_path = "ToxicModel" + str(fold_id) + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    
    # training on given fold data
    hist = model.fit(train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=50, batch_size=batch_size, shuffle=True,
        callbacks=[early_stopping, model_checkpoint])
    
    # get the minimal validation log loss on this fold
    bst_val_score = min(hist.history['val_loss'])
    model.load_weights(bst_model_path)

    # return the model with best weight, best fold-val score
    return model, bst_val_score

def train_folds(X, y, fold_count, batch_size, get_model_func):
    fold_size = len(X) // fold_count
    models = []
    score = 0
    
    # split the whole dataset to `fold_count` fold, and train our model on each fold
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        # Generate the train/val data on fold i
        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = X[fold_start:fold_end]
        val_y = y[fold_start:fold_end]
    
        print("Training on fold #", fold_id)
        model, bst_val_score = _train_model_by_logloss(get_model_func(), batch_size, train_x, train_y, val_x, val_y, fold_id)
        score += bst_val_score
        models.append(model)
    return models, score / fold_count
models, val_loss = train_folds(train_data, train_labels, 2, 256, get_text_cnn)
your_batch_size = 256

def get_your_model():
    filter_nums = 120
    dense_units = 72
    output_units = 6
    
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,trainable=False,)(input_layer)
        
    conv_0 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")(embedding_layer)
    conv_1 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")(embedding_layer)
    conv_2 = Conv1D(filter_nums, 4, kernel_initializer="normal", padding="valid", activation="relu")(embedding_layer)

    maxpool_0 = GlobalMaxPooling1D()(conv_0)
    maxpool_1 = GlobalMaxPooling1D()(conv_1)
    maxpool_2 = GlobalMaxPooling1D()(conv_2)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2])
    h1 = Dense(units=dense_units, activation="relu")(merged_tensor)
    output = Dense(units=output_units, activation='sigmoid')(h1)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
models, val_loss = train_folds(train_data, train_labels, 2, your_batch_size, get_your_model)
#test_data = test_df
CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
submit_path_prefix = "ToxicNN-" + str(MAX_SEQUENCE_LENGTH) 

print("Predicting testing results...")
test_predicts_list = []
for fold_id, model in enumerate(models):
    test_predicts = model.predict(test_data, batch_size=256, verbose=1)
    test_predicts_list.append(test_predicts)

# merge each folds' predictions by averaging
test_predicts = np.zeros(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts += fold_predict
test_predicts /= len(test_predicts_list)

# create the submission file
test_ids = test_df["id"].values
test_ids = test_ids.reshape((len(test_ids), 1))
test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
test_predicts["id"] = test_ids
test_predicts = test_predicts[["id"] + CLASSES]
submit_path = submit_path_prefix + "-Loss{:4f}.csv".format(val_loss)
test_predicts.to_csv(submit_path, index=False)


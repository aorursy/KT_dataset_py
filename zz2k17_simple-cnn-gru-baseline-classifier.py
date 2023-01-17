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
import keras
# verify GPU acceleration
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# functions for reading in embedding data and
# tokenizing and processing sequences with padding and
# function for plotting model accuracy and loss

import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# vectorizer and sequence function
# takes in raw text and labels
# params for max sequence length and max words
# default arg for Shuffle=True to randomise data
# returns tokenizer object. x_train,y_train, x_val,y_val
def tokenize_and_sequence(full_data_set, texts, labels, max_len, max_words, validation_samples, shuffle=True):
    #initialise tokenizer with num_words param
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(full_data_set)
    # convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    # generate work index
    word_index = tokenizer.word_index
    # print top words count
    print('{} of unique tokens found'.format(len(word_index)))
    # pad sequences using max_len param
    data = pad_sequences(sequences, maxlen=max_len)
    # convert list of labels into numpy array
    labels = np.asarray(labels)
    # print shape of text and label tensors
    print('data tensor shape: {}\nlabel tensor shape:{}'.format(data.shape, labels.shape))

    # shuffle data=True as labels are ordered
    # randomise data to vary class distribution
    if shuffle:
        # get length of data sequence and create array
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        # shuffle data and labels
        data = data[indices]
        labels = labels[indices]
    else:
        pass

    # split training data into training and validation splits
    # split using validation length
    # validation split
    x_val = data[:validation_samples]
    y_val = labels[:validation_samples]
    # training split
    x_train = data[validation_samples:]
    y_train = labels[validation_samples:]

    # return tokenizer, word_index, training and validation data
    return tokenizer, word_index, x_train, y_train, x_val, y_val


# function to lpad pretrained glove embeddings
# takes in embedding dim for variable embedding sizes
# and base directory as well as txt file
# embedding dim should match the file name dimension
# and max words and word_index for embedding features
def load_glove(base_directory, f_name, max_words, word_index, embedding_dim=None):
    # check file name ends in .txt
    # read file name embedding value if not specified
    if f_name[-4:] == '.txt':
        # check embedding value
        if embedding_dim is not None:
            dim = f_name[-8:-5]
            dim = int(dim)
            embedding_dim = dim
        else:
            # assuming dimension is not none for manual input
            pass
        # continue

        # create embedding dictionary
        embeddings_index = {}
        # open embeddings file
        try:
            f = open(os.path.join(base_directory, f_name))
            # iterate over lines and split on individual words
            # split coefficient of word values
            # map words and coefficients to embeddings dictionary
            for line in f:
                values = line.split() # returns list of [word, coeff]
                word = values[0] # gets first list element
                coeff = np.asarray(values[1:], dtype='float32')  # slice coefficiennt value array from remainder of list
                # assign mapping to dictionary
                embeddings_index[word] = coeff
            f.close()
        except IOError:
            print('cannot read file. check file paths')

        # prepare glove word-embedding matrix
        # create empty embedding tensor
        embedding_matrix = np.zeros((max_words,embedding_dim ))
        # map the top words of the data into the glove embedding matrix
        # words not found from the data in glove will be zeroed
        for word, i in word_index.items():
            if i < max_words:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        # return embedding matrix
        return embedding_matrix


# function to visualise keras model history metrics
# function takes in acc, val_acc, loss, val_loss for model params
# range is defined by epochs in range len(acc)

import matplotlib.pyplot as plt

def plot_training_and_validation(acc, val_acc, loss, val_loss):
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

# end
import pandas as pd
# set file paths for training and test files
# base directory
toxic_base = '../input/jigsaw-toxic-comment-classification-challenge/'
toxic_train = toxic_base + 'train.csv'
toxic_test = toxic_base + 'test.csv'
# load train and test data into DataFrames
train_df = pd.read_csv(toxic_train)
test_df = pd.read_csv(toxic_test)
# verify and inspect data
# training data
train_df.head()
print(train_df.shape)
# test data
test_df.head()
test_df.shape
# extract label columns
cols = list(train_df.columns)
print(type(cols))
print(list(cols))
# remove id and comment_text columns
cols = cols[2:]
print(cols)
# describe data and check for any null or missing values, and the spread of labels
train_df.describe()
# check for null values in labels in training data
train_df.isnull().any()
# check for null values in test data
test_df.isnull().any()
# view spread of classess across the training data
for label in cols:
    print('label: {}'.format(label))
    for x in [0, 1]:
        print('value: {}, total:{}'.format(x, train_df[label].eq(x).sum()))

# split class labels into a y array
y_train = train_df[cols].values
# verify labels are arrays of 6 values
y_train[0]
# convert to numpy array
y_train = np.asarray(y_train)
# verify matrix shape
y_train.shape
# split comment_text from training and test data
comment = 'comment_text'
# extract Series object from DataFrames for train and test 
sentences_train = train_df[comment]
sentences_test = test_df[comment]
# transform Series into list of text values
x_train = list(sentences_train)
x_test = list(sentences_test)
# verify train and test text
print(x_train[0], '\n')
print(x_test[0])
# check training data sample size
print(len(x_train))
# get the value of 10% of the traning data
print(len(x_train) // 10)
validation_samples = int((len(x_train) // 10))
validation_samples
# define max sequence length and total dictionary words
max_len = 100
max_words = 10000
# Vectorize training data and return tokenizer and word_index as well as validation splits
tokenizer, word_index, X_train, Y_train, x_val, y_val = tokenize_and_sequence(
    x_train, y_train, max_len=max_len, max_words=max_words, validation_samples=validation_samples, shuffle=False)
# verify train and validation text and labels
print('training:',X_train.shape, Y_train.shape, '\nvalidation:', x_val.shape, y_val.shape)
# define directory paths for glove embeddings
glove_dir = '../input/glove6b200d/'
# glove file name
glove_file = 'glove.6B.200d.txt'
# define embedding dimension value to match the glove200d file
embedding_dim = 200
# load in glove embedding using custom function from earlier
# function takes as input the raw file, word_index returned from the tokenizer and max_words
glove_embedding_200d = load_glove(
    glove_dir, glove_file, max_words=max_words, word_index=word_index, embedding_dim=embedding_dim)
# verify embeddings loaded correctly
glove_embedding_200d.shape
# import layers
from keras.layers import Input, Embedding, GRU, LSTM, MaxPooling1D, GlobalMaxPool1D
from keras.layers import Dropout, Dense, Activation, Flatten,Conv1D, SpatialDropout1D
from keras.models import Sequential
from keras.optimizers import RMSprop 
# import AUC ROC metrics from sklearn
from sklearn.metrics import roc_auc_score
# define model architecture
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(4))
model.add(GRU(64, dropout=0.1, recurrent_dropout=0.5)) # defaults inclide tanh activation
model.add(Dense(6, activation='sigmoid'))
model.summary()
# load pre-trained Glove embeddings in the first layer
model.layers[0].set_weights([glove_embedding_200d])
# freeze embedding layer weights
model.layers[0].trainable = False
# compile model with adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit model and train on training data and validate on validation samples
# train for 5 epochs to establish baseline overfitting model
# saves results to histroy object
history = model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))
# save model
model.save('cnn_gru_200d.h5')
# define plotting metrics
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
# plot model training and validation accuracy and loss
plot_training_and_validation(acc, val_acc, loss, val_loss)
y_hat = model.predict(x_val)
# print auc roc score
"{:0.2f}".format(roc_auc_score(y_val, y_hat)*100.0)
# verify length of training and test data
print(len(x_train), len(x_test))
# create validation split of 10%
validation_split = len(x_train) // 10
print(validation_split)
# define max_len and max_words
max_len = 50
max_words = 20000
full_tokenized = x_train + x_test
# repeat vectorization process using our custom function
# Vectorize training data and return tokenizer and word_index as well as validation splits
# tokenize on x_train AND x_test using new parameter to capture as many words as possible
tokenizer, word_index, X_train, Y_train, x_val, y_val = tokenize_and_sequence(full_tokenized,
    x_train, y_train, max_len=max_len, max_words=max_words, 
    validation_samples=validation_split, shuffle=False)
# define directory paths for glove embeddings
glove_dir = '../input/glove6b200d/'
# glove file name
glove_file = 'glove.6B.200d.txt'
# define embedding dimension value to match the glove200d file
embedding_dim = 200
# load in glove embedding using custom function from earlier
# function takes as input the raw file, word_index returned from the tokenizer and max_words
glove_embedding_200d = load_glove(
    glove_dir, glove_file, max_words=max_words, word_index=word_index, embedding_dim=embedding_dim)
# verify embeddings loaded correctly
glove_embedding_200d.shape
# import AUC ROC metrics from sklearn
from sklearn.metrics import roc_auc_score

# define class for ROC AUC callback with simple name modifications
# credit to https://www.kaggle.com/yekenot
class roc_auc_validation(keras.callbacks.Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

# import keras layers 
from keras.layers import Input, Embedding, GRU, LSTM, MaxPooling1D, GlobalMaxPool1D, CuDNNGRU, CuDNNLSTM
from keras.layers import Dropout, Dense, Activation, Flatten,Conv1D, Bidirectional, SpatialDropout1D, BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop 
# define model architecture
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(SpatialDropout1D(0.2)) # add spatial dropout
model.add(Conv1D(64, 3, activation='relu')) # increase kernel size to 5 # change to 3 to test
model.add(MaxPooling1D(4))
model.add(BatchNormalization()) # add batch normalization
model.add(Dropout(0.1))
# modify to CuDNNGRU
#model.add(GRU(64, dropout=0.1, recurrent_dropout=0.5)) # defaults inclide tanh activation
model.add(CuDNNGRU(64)) # does not have a dropout or recurrent dropout param
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(6, activation='sigmoid'))
model.summary()
# Baseline++ with more units
# this model performs the best with the highest training accuracy of loss: 0.0378 - acc: 0.9857 
# - val_loss: 0.0763 - val_acc: 0.9759 # ROC-AUC - epoch: 20 - score: 0.951627
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(SpatialDropout1D(0.2)) # add spatial dropout
model.add(Conv1D(128, 5, activation='relu')) # increase kernel size to 5
model.add(MaxPooling1D(4))
model.add(BatchNormalization()) # add batch normalization
model.add(Dropout(0.1))

# modify to CuDNNGRU and double units to 128
#model.add(GRU(64, dropout=0.1, recurrent_dropout=0.5)) # defaults inclide tanh activation
model.add(Bidirectional(CuDNNGRU(128))) # does not have a dropout or recurrent dropout param
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(6, activation='sigmoid'))
model.summary()
# Baseline++ with 2 CNN layers
# less performant compared to baseline++
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(SpatialDropout1D(0.2)) # add spatial dropout
# CNN layers
model.add(Conv1D(256, 5, activation='relu')) # increase kernel size to 5
model.add(MaxPooling1D(4))
model.add(BatchNormalization()) # add batch normalization
model.add(Dropout(0.1))
# second CNN 
model.add(Conv1D(128, 7, activation='relu')) # increase kernel size to 5
model.add(MaxPooling1D(4))
model.add(BatchNormalization()) # add batch normalization
model.add(Dropout(0.1))
# modify to CuDNNGRU and double units to 128
#model.add(GRU(64, dropout=0.1, recurrent_dropout=0.5)) # defaults inclide tanh activation
model.add(Bidirectional(CuDNNGRU(64))) # does not have a dropout or recurrent dropout param
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(6, activation='sigmoid'))
model.summary()
# highest performing architecture
# baseline++ with shorter convolution kernel of 3
# define model architecture
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(SpatialDropout1D(0.2)) # add spatial dropout
model.add(Conv1D(64, 3, activation='relu')) # increase kernel size to 5 # change to 3 to test
model.add(MaxPooling1D(4))
model.add(BatchNormalization()) # add batch normalization
model.add(Dropout(0.1))
# modify to CuDNNGRU
#model.add(GRU(64, dropout=0.1, recurrent_dropout=0.5)) # defaults inclide tanh activation
model.add(CuDNNGRU(64)) # does not have a dropout or recurrent dropout param
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(6, activation='sigmoid'))
model.summary()
# define callbacks
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
# initialise customer roc callback
roc_callback = roc_auc_validation(validation_data=(x_val, y_val), interval=1)
# define early stopping and reduce lr callbacks
callback_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=1),
                 keras.callbacks.ModelCheckpoint(filepath='baseline_plus_complex.h5', monitor='val_loss',
                                                 save_best_only=True)]
# add roc to callbacks list
callback_list.append(roc_callback)
callback_list
# load pre-trained Glove embeddings in the first layer
model.layers[0].set_weights([glove_embedding_200d])
# freeze embedding layer weights
model.layers[0].trainable = False
# compile model with adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit model and train on training data and validate on validation samples
# train for 5 epochs to establish baseline overfitting model
# saves results to histroy object
history = model.fit(X_train, Y_train, epochs=20, batch_size=256, callbacks=callback_list,validation_data=(x_val, y_val))
# tokenize and pad test data
X_test = tokenizer.texts_to_sequences(x_test)
X_test = pad_sequences(X_test, maxlen=max_len)
# verify test data shape
X_test.shape
# test model on submission as reading in test_labels fails
y_hat = model.predict(X_test, batch_size=256)
submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_hat
submission.to_csv('submission_baseline_plus_best_k3.csv', index=False)

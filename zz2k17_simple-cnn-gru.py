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
# functions for reading in embedding data and
# tokenizing and processing sequences with padding and
# function for plotting model accuracy and loss
# modify line.split to line.split(" ") as 300D contains spaces

import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# vectorizer and sequence function
# takes in raw text and labels
# params for max sequence length and max words
# default arg for Shuffle=True to randomise data
# returns tokenizer object. x_train,y_train, x_val,y_val
def tokenize_and_sequence(texts, labels, max_len, max_words, validation_samples, shuffle=True):
    #initialise tokenizer with num_words param
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
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
                values = line.split(" ") # returns list of [word, coeff]
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
base_dir ='../input'
print(base_dir)
# list files in current directory
print(os.listdir(base_dir))
# set train and test data set paths
train_path = os.path.join(base_dir, 'train.csv')
test_path = os.path.join(base_dir, 'test.csv')
print(train_path, test_path)
# set embedding file path
print(os.listdir(os.path.join(base_dir, 'embeddings')))
glove_file = 'glove.840B.300d.txt'
base_embedding_dir = os.path.join(base_dir, 'embeddings/glove.840B.300d')
print(os.path.join(base_embedding_dir, glove_file))
# load data into DataFrames
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
# verify and inspect DataFrames
train_df.head()
test_df.head()
# simple statistics of training data
train_df.info()
train_df.shape
# check for null values in labels in training data
train_df.isnull().any()
# define variable to index data frame question column
questions = 'question_text'
# find maximum sequence length of questions
np.max(train_df[questions].apply(lambda x: len(x.split())))
# split labels from training data into numpy array
y_train = train_df['target'].values
y_train = np.asarray(y_train)
print(type(y_train))
# verify label array shape
y_train.shape
# inspect label
y_train[0]
# extract questions from Series objects of train and test data
questions_train = train_df[questions]
questions_test = test_df[questions]
# transforms Series into lists
x_train = list(questions_train)
x_test = list(questions_test)
# verify and inspect data
print(x_train[0], y_train[0], type(x_train), len(x_train))
print(x_test[0], len(x_test))
# define a 10% validation split from training data
validation_samples = int((len(x_train) // 10))
# verify 90:10 split
print(len(x_train), validation_samples)
# define max sequence length and total dictionary words
max_len = 100
max_words = 10000
# Vectorize training data and return tokenizer and word_index as well as validation splits
tokenizer, word_index, X_train, Y_train, x_val, y_val = tokenize_and_sequence(
    x_train, y_train, max_len=max_len, max_words=max_words, validation_samples=validation_samples, shuffle=False)
# verify train and validation text and labels
print('training:',X_train.shape, Y_train.shape, '\nvalidation:', x_val.shape, y_val.shape)
# define embedding dimension
embedding_dim = 300
# load in glove embedding using custom function from earlier
# function takes as input the raw file, word_index returned from the tokenizer and max_words
glove_embedding_300d = load_glove('../input/embeddings/glove.840B.300d/', glove_file, max_words=max_words, word_index=word_index, embedding_dim=embedding_dim)
# *error in loading* needs investigation 
# * 300D needs line.split(' ') compared to smaller dimensions
# verify embeddings loaded correctly
glove_embedding_300d.shape
# import layers
from keras.layers import Input, Embedding, GRU, LSTM, MaxPooling1D, GlobalMaxPool1D, CuDNNGRU
from keras.layers import Dropout, Dense, Activation, Flatten, Conv1D, SpatialDropout1D
from keras.models import Sequential
# import AUC ROC metrics from sklearn
from sklearn.metrics import roc_auc_score
# define model architecture
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Conv1D(64, 3, activation='relu'))
model.add(SpatialDropout1D(0.2))
model.add(MaxPooling1D(4))
model.add(CuDNNGRU(64))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# load pre-trained Glove embeddings in the first layer
model.layers[0].set_weights([glove_embedding_300d])
# freeze embedding layer weights
model.layers[0].trainable = False
# compile model with adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit model and train on training data and validate on validation samples
# train for 5 epochs to establish baseline overfitting model
# saves results to histroy object
history = model.fit(X_train, Y_train, epochs=5, batch_size=128, validation_data=(x_val, y_val))
# save model
model.save('cnn_cudnngru_300d.h5')
# define plotting metrics
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
# plot model training and validation accuracy and loss
plot_training_and_validation(acc, val_acc, loss, val_loss)
# further enhancements to be made and model checkpointing
# plots show model performs the best at epoch 3
y_hat = model.predict(x_val)
# print auc roc score
"{:0.2f}".format(roc_auc_score(y_val, y_hat)*100.0)
# we first need to tokenize and pad the raw text from the test data
sequences_test = tokenizer.texts_to_sequences(x_test)
test_data = pad_sequences(sequences_test, maxlen=max_len)
test_data.shape
# verify test data sample
test_data[0]
# iterate over test sequences and predict y_hat values
y_hat = model.predict(test_data)
y_hat.shape
y_hat
predictions = (np.array(y_hat) > 0.5).astype(np.int)
# create dataframe for precitions
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": predictions.flatten()})
submit_df.head()
# save as csv for submission
# submit_df.to_csv('submission.csv', index=False)
# functions for reading in embedding data and
# tokenizing and processing sequences with padding and
# function for plotting model accuracy and loss
# modify line.split to line.split(" ") as 300D contains spaces

import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# vectorizer and sequence function
# takes in raw text and labels
# params for max sequence length and max words
# default arg for Shuffle=True to randomise data
# returns tokenizer object. x_train,y_train, x_val,y_val
def tokenize_and_sequence(full_texts, texts, labels, max_len, max_words, validation_samples, shuffle=True):
    #initialise tokenizer with num_words param
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(full_texts)
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
                values = line.split(" ") # returns list of [word, coeff]
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
full_texts = x_train + x_test
max_len = 100
max_words = 10000
# vectorize training data
# Vectorize training data and return tokenizer and word_index as well as validation splits
tokenizer, word_index, X_train, Y_train, x_val, y_val = tokenize_and_sequence(
    full_texts, x_train, y_train, max_len=max_len, max_words=max_words, validation_samples=validation_samples, shuffle=False)
# verify train and validation text and labels
print('training:',X_train.shape, Y_train.shape, '\nvalidation:', x_val.shape, y_val.shape)
# define embedding dimension
embedding_dim = 300
# load in glove embedding using custom function from earlier
# function takes as input the raw file, word_index returned from the tokenizer and max_words
glove_embedding_300d = load_glove('../input/embeddings/glove.840B.300d/', glove_file, max_words=max_words, word_index=word_index, embedding_dim=embedding_dim)
# * 300D needs line.split(' ') compared to smaller dimensions
# verify embeddings loaded correctly
glove_embedding_300d.shape
# import keras layers
import keras
# define custom ROC callback
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
model.add(Conv1D(64, 5, activation='relu')) # increase kernel size to 5
model.add(MaxPooling1D(4))
model.add(BatchNormalization()) # add batch normalization
model.add(Dropout(0.1))
# modify to CuDNNGRU
#model.add(GRU(64, dropout=0.1, recurrent_dropout=0.5)) # defaults inclide tanh activation
model.add(CuDNNGRU(64)) # does not have a dropout or recurrent dropout param
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# define callbacks
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
# initialise customer roc callback
roc_callback = roc_auc_validation(validation_data=(x_val, y_val), interval=1)
# define early stopping and reduce lr callbacks
callback_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=1),
                 keras.callbacks.ModelCheckpoint(filepath='baseline_plus_.h5', monitor='val_loss',
                                                 save_best_only=True)]
# add roc to callbacks list
callback_list.append(roc_callback)
callback_list
# load pre-trained Glove embeddings in the first layer
model.layers[0].set_weights([glove_embedding_300d])
# freeze embedding layer weights
model.layers[0].trainable = False
# compile model with adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit model and train on training data and validate on validation samples
# train for 5 epochs to establish baseline overfitting model
# saves results to histroy object
history = model.fit(X_train, Y_train, epochs=20, batch_size=512, callbacks=callback_list,validation_data=(x_val, y_val))
# evaluate model on test set and submit results
# we first need to tokenize and pad the raw text from the test data
sequences_test = tokenizer.texts_to_sequences(x_test)
test_data = pad_sequences(sequences_test, maxlen=max_len)
# iterate over test sequences and predict y_hat values
y_hat = model.predict(test_data)
predictions = (np.array(y_hat) > 0.5).astype(np.int)
# create dataframe for precitions
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": predictions.flatten()})
submit_df.to_csv('submission.csv', index=False)

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import csv

import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
tweets = [] 

labels = []

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]



train_with_stopwords = False



with open("../input/nlp-getting-started/train.csv") as csvfile: #iterate in csv files and extract tweets and relevant labels

    reader = csv.reader(csvfile, delimiter=',')

    next(reader) #skip table header

    for row in reader:

        labels.append(int(row[4]))

        url = re.compile(r'https?://\S+|www\.\S+') #remove url if any in tweet message.

        tweet = url.sub(r'',row[3])

        if train_with_stopwords == True:

            for word in stopwords:

                token = " " + word + " "

                tweet = tweet.replace(token, " ")

        tweets.append(tweet)

print(f'Total training set = {len(tweets)} tweets')

print(f'Total training set = {len(labels)} tweets')
#for total number of training size = 7613 tweets, first let's train on 6,000 samples and validate on 1,613 samples. After fine tuning model until it reach desired performance, later we'll train on total training set.

train_tweets , val_tweets, train_labels , val_labels = train_test_split(tweets,labels, train_size = 6000 , stratify = labels, random_state=42)

print(f'Training set size = {len(train_tweets)}')

print(f'Validation set size = {len(val_tweets)}')
embedding_dim = 100 #to match with glove model we import

max_length = 50

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"
tokenizer = Tokenizer()

tokenizer.fit_on_texts(tweets)

word_index = tokenizer.word_index

vocab_size = len(word_index) #set vocab size to maximum



train_sequences = tokenizer.texts_to_sequences(train_tweets)

train_padded = pad_sequences(train_sequences, padding = padding_type, truncating = trunc_type, maxlen = max_length)



val_sequences = tokenizer.texts_to_sequences(val_tweets)

val_padded = pad_sequences(val_sequences, padding = padding_type, truncating = trunc_type, maxlen = max_length)
sns.set(style = 'darkgrid',

       context = 'notebook',

       palette = 'muted',

       )

sns.despine(left=True)
df = pd.read_csv('../input/nlp-getting-started/train.csv')

df['length'] = df['text'].str.len() # compute tweet message count

df['word_counts'] = df['text'].str.split().apply(len)

df.head()
print(f"Training dataset shape\n {df.shape[0]} rows, {df.shape[1]} columns")

print('\nTotal missing values')

print(df.isnull().sum())

df = df.drop(columns = ['keyword','location'])
word_count = tokenizer.word_counts

word_count_sorted = {k:v for k,v in sorted(word_count.items(),reverse=True,key=lambda item: item[1])}

most_common_word = []

most_common_word_count = []

i = 0

for k,v in word_count_sorted.items():

    if i == 20:

        break

    most_common_word.append(k)

    most_common_word_count.append(v)

    i += 1
f, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.despine(left=True)



#Class distribution graph

ax1 = sns.countplot(x='target',data=df, ax = axes[0,0])

ax1.set_title('Class distribution',size = 15, y = 1.02)

ax1.set_xlabel('Target')

ax1.set_ylabel('Number of observation')



#Distribution of tweet message length of Real and Fake Disaster

ax2 = sns.kdeplot(df[df['target'] == 1].length ,shade = True, label = 'Real Disaster!' ,color ='b', ax=axes[0,1])

ax3 = sns.kdeplot(df[df['target'] == 0].length , shade = True, label = 'Fake!', color = 'r', ax=axes[0,1])

ax2.set_xlabel('Tweet message length (number of characters)')

ax2.set_title('Distribution of tweet message length of Real and Fake Disaster',size = 15, y = 1.02)



#Distribution of Number of words per tweets of Real and Fake Disaster

ax3 = sns.kdeplot(df[df['target'] == 1].word_counts ,shade = True, label = 'Real Disaster!' ,color ='b', ax=axes[1,0])

ax4 = sns.kdeplot(df[df['target'] == 0].word_counts , shade = True, label = 'Fake!', color = 'r', ax=axes[1,0])

ax3.set_xlabel('Number of words per tweets')

ax3.set_title('Distribution of Number of words per tweets of Real and Fake Disaster',size = 15, y = 1.02)



#Word counts

ax5 = sns.barplot(x=most_common_word,y=most_common_word_count,

            palette = sns.cubehelix_palette(n_colors=20,rot=-.3,reverse=True), ax=axes[1,1])

ax5.set_xticklabels(ax5.get_xticklabels(),

                  rotation=90);

ax5.set_ylabel('Number of observation')

ax5.set_title('20 Most frequent words',size = 15, y = 1.02)



plt.tight_layout(pad=2.0);
#Special thanks to Laurence Moroney for his hosted site for easier download



#This block of code simply map our words in tokenizer to 100-dimensional vectors with pretrained weight

embeddings_index = {};

with open('../input/standford-glove/glove.6B.100d.txt') as f:

    for line in f:

        values = line.split();

        word = values[0];

        coefs = np.asarray(values[1:], dtype='float32');

        embeddings_index[word] = coefs;



embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word);

    if embedding_vector is not None:

        embeddings_matrix[i] = embedding_vector;
import matplotlib.image  as mpimg

import matplotlib.pyplot as plt



#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

def plot_lr_lc(history):

    acc=history.history['accuracy']

    val_acc=history.history['val_accuracy']

    loss=history.history['loss']

    val_loss=history.history['val_loss']

    

    epochs=range(len(acc)) # Get number of epochs

    

    #------------------------------------------------

    # Plot learning rate vs loss to find optimal learning rate

    #------------------------------------------------

    plt.semilogx(history.history["lr"], history.history["loss"])

    plt.axis([1e-5, 10, 0, 1])

    plt.xlabel('Learning Rate')

    plt.ylabel('Loss')

    plt.title(str(history))    

    plt.figure()

    

    #------------------------------------------------

    # Plot training and validation accuracy per epoch

    #------------------------------------------------

    plt.plot(epochs, acc, 'r')

    plt.plot(epochs, val_acc, 'b')

    plt.title('Training and validation accuracy')

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend(["Accuracy", "Validation Accuracy"])

    plt.axis([0,80,0,1])

    plt.figure()

    

    #------------------------------------------------

    # Plot training and validation loss per epoch

    #------------------------------------------------

    plt.plot(epochs, loss, 'r')

    plt.plot(epochs, val_loss, 'b')

    plt.title('Training and validation loss')

    plt.xlabel("Epochs")

    plt.ylabel("Loss")

    plt.legend(["Loss", "Validation Loss"])

    plt.axis([0,80,0,5])

    plt.figure()





# Expected Output

# A chart where the validation loss does not increase sharply!
tf.keras.backend.clear_session()

tf.random.set_seed(51)

np.random.seed(51)



#hyperparameter to tune

l2_weight = 0.01

dropout_rate = 0.3

initial_lr = 0.001



lr_schedule = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: initial_lr * 10**(epoch/20))



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1,embedding_dim,input_length = max_length, weights = [embeddings_matrix], trainable=False),

    tf.keras.layers.Conv1D(16, 5, activation='relu'),

    tf.keras.layers.Dropout(dropout_rate),

    tf.keras.layers.MaxPooling1D(pool_size=4),

    tf.keras.layers.Bidirectional(LSTM(32,dropout = dropout_rate)),

    tf.keras.layers.Dense(8, activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l2_weight)),

    tf.keras.layers.Dense(1, activation='sigmoid')

])





model.compile(loss='binary_crossentropy', 

              optimizer = tf.keras.optimizers.SGD(lr=initial_lr ,momentum=0.9) ,

              metrics=['accuracy']

             )

model.summary()
num_epochs = 50

history = model.fit(train_padded, np.array(train_labels), epochs=num_epochs, 

                    validation_data=(val_padded, np.array(val_labels)), 

                    verbose=1,

                   callbacks = [lr_schedule]

                   )
plot_lr_lc(history)
tf.keras.backend.clear_session()

tf.random.set_seed(51)

np.random.seed(51)

#hyperparameter to tune

l2_weight = 0.01

dropout_rate = 0.3



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1,embedding_dim,input_length = max_length, weights = [embeddings_matrix], trainable=False),

    tf.keras.layers.Conv1D(16, 5, activation='relu'),

    tf.keras.layers.Dropout(dropout_rate),

    tf.keras.layers.MaxPooling1D(pool_size=4),

    tf.keras.layers.Bidirectional(LSTM(32,dropout = dropout_rate)),

    tf.keras.layers.Dense(8, activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l2_weight)),

    tf.keras.layers.Dense(1, activation='sigmoid')

])





model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.SGD(lr=3e-2 ,momentum=0.9) ,metrics=['accuracy'])

model.summary()
num_epochs = 100





#Early stop the training if there's no improvement in model performance for 20 epochs

early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)





#Reduce model learning rate if validation loss reach plateau

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=20, min_lr=0.001)



#Checkpoint callback for model with lowest validation accuracy

checkpoint_filepath = 'checkpoint'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_filepath,

    save_weights_only=True,

    monitor='val_acc',

    save_best_only=True)





history = model.fit(train_padded, np.array(train_labels), epochs=num_epochs, 

                    validation_data=(val_padded, np.array(val_labels)), 

                    verbose=1,

                    callbacks= [early,model_checkpoint_callback]

                   )
#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------



acc=history.history['accuracy']

val_acc=history.history['val_accuracy']

loss=history.history['loss']

val_loss=history.history['val_loss']

# lr = history.history['lr']



epochs=range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot lr change vs epoch

#------------------------------------------------

# plt.plot(epochs,lr)

# plt.title('lr change vs epoch')

# plt.xlabel("Epochs")

# plt.ylabel("learning rate")

# plt.figure()



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, 'r')

plt.plot(epochs, val_acc, 'b')

plt.title('Training and validation accuracy')

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend(["Accuracy", "Validation Accuracy"])

plt.axis([0,len(history.history['loss']),.5,1])

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r')

plt.plot(epochs, val_loss, 'b')

plt.title('Training and validation loss')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend(["Loss", "Validation Loss"])

plt.axis([0,len(history.history['loss']),0,1])

plt.figure()
model.load_weights(r'./' + checkpoint_filepath)
#load test set with same preprocessing steps as training set

tweets_test = [] 

ids = []

with open("../input/nlp-getting-started/test.csv") as csvfile: #iterate in csv files and extract tweets and relevant labels

    reader = csv.reader(csvfile, delimiter=',')

    next(reader) #skip table header

    for row in reader:

        ids.append(int(row[0]))

        url = re.compile(r'https?://\S+|www\.\S+') #remove url if any in tweet message.

        tweet = url.sub(r'',row[3])

        tweets_test.append(tweet)

print(f'Total training set = {len(tweets_test)} tweets')

print(f'Total training set = {len(ids)} tweets')
test_sequences = tokenizer.texts_to_sequences(tweets_test)

test_padded = pad_sequences(test_sequences, padding = padding_type, truncating = trunc_type, maxlen = max_length)
y_predict = model.predict(test_padded)

y_predict = np.round(y_predict,0).astype(int).flatten()
submission = pd.DataFrame({'id' : ids, 'target' : y_predict})

print(submission.head())

submission.to_csv('submission.csv',index=False)
train_tweets_full = np.vstack((train_padded,val_padded))

train_labels_full = np.concatenate((np.array(train_labels),np.array(val_labels)))
num_epochs = 30



history = model.fit(train_tweets_full, train_labels_full, epochs=num_epochs, 

                    verbose=1,

                   )
y_predict_full = model.predict(test_padded)

y_predict_full = np.round(y_predict_full,0).astype(int).flatten()
submission_full = pd.DataFrame({'id' : ids, 'target' : y_predict_full})

print(submission_full.head())

submission_full.to_csv('submission_full.csv',index=False)
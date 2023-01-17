# !pip install tensorflow==2.0.0-alpha0

# kaggle default imports

import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import seaborn as sns

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

import time

from wordcloud import WordCloud, STOPWORDS

from IPython.core.display import display, HTML

import plotly.graph_objects as go

import re

# Natural Language Tool Kit

import nltk

# nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from collections import Counter

import cufflinks as cf

cf.go_offline()
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub =  pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



train_df.head()
print(f"Train data shape {train_df.shape}")

print(f"Test data shape {test_df.shape}")
missing = train_df.isnull().sum()

missing[missing>0].sort_values(ascending=False).iplot(kind='bar',

                                                      title='Null values present in train Dataset',

                                                      color=['blue'])
train_df.target.value_counts().iplot(kind='bar',

                                     text=['Fake', 'Real'],

                                     title='Comparing Tweet is a real disaster (1) or not (0)',

                                     color=['green'])
counts_train = train_df.target.value_counts(sort=False)

labels = counts_train.index

values_train = counts_train.values



data = go.Pie(labels=labels, values=values_train, pull=[0.03, 0])

layout = go.Layout(title="Comparing Tweet is a real disaster (1) or not (0) in %")

fig = go.Figure(data=[data], layout=layout)

fig.update_traces(hole=.3, hoverinfo="label+percent+value")

fig.update_layout(

    # Add annotations in the center of the donut pies

    annotations = [dict(text='Train', x=0.5, y=0.5, font_size=20, showarrow=False)]

)

fig.show()
train_df['length'] = train_df['text'].apply(len)



data = [

    go.Box(

        y=train_df[train_df['target']==0]['length'],

        name='Fake'

    ),

    go.Box(

        y=train_df[train_df['target']==1]['length'],

        name='Real'

    )

]

layout = go.Layout(title='Comparison of text length in Tweets')

fig = go.Figure(data=data, layout=layout)

fig.show()
print(f"There are {train_df.keyword.nunique()} in the train set") # Total number of unique keywords



train_df.keyword.value_counts()[:20].iplot(kind='bar', title='Top 20 keywords in text', color='red')
train_df.location.value_counts()[:20].iplot(kind='bar', title='Top 20 locations in tweet')
STOPWORDS.add('https')



def plot_world(text):

    comment_words = ' '

    stopwords = set(STOPWORDS)

    for val in text:

        # typecast val into string

        val = str(val)

        # split the tokens

        tokens = val.split()

        for i in range(len(tokens)):

            # converts each token into lowercase

            tokens[i] = tokens[i].lower()

        for words in tokens:

            comment_words = comment_words + words + ' '

            

    wordcloud = WordCloud(width = 5000,

                          height = 4000,

                          background_color = 'black',

                          stopwords = stopwords,

                          min_font_size = 10).generate(comment_words)

    # plot the WordCloud image

    plt.figure(figsize=(12,12), facecolor='k', edgecolor='k')

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.tight_layout(pad=0)

    plt.show()
# text = train_df.text.values

# plot_world(text)
# How many http words has this text?

train_df.loc[train_df['text'].str.contains('http')].target.value_counts()
pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')



def remove_html(text):

    no_html= pattern.sub('',text)

    return no_html



# Remove all text that start with html

train_df['text'] = train_df['text'].apply(lambda x: remove_html(x))
# lets check if this clean works

print(train_df.loc[train_df['text'].str.contains('http')].target.value_counts())



# remove all text that start with html in test

test_df['text'] = test_df['text'].apply(lambda x: remove_html(x))
# now remove stopwords, change to lowercase



def clean_text(text):

    text = re.sub('[^a-zA-Z]', ' ', text)

    text = text.lower()

    # split to array (default delimiter is " ")

    text = text.split()

    text = [w for w in text if not w in set(stopwords.words('english'))]

    text = ' '.join(text)

    return text
# apply text cleaning

train_df['text'] = train_df['text'].apply(lambda x : clean_text(x))

test_df['text'] = test_df['text'].apply(lambda x : clean_text(x))
# How many unique words have this text



def counter_word(text):

    count = Counter()

    for i in text.values:

        for word in i.split():

            count[word] += 1

    return count

text_values = train_df['text']



counter = counter_word(text_values)

print(f"The len of words is: {len(counter)}")

list(counter.items())[:10]
# The maximum number of words to be used. (most frequent)



vocab_size = len(counter)

embedding_dim = 32



# Max number of words in each complaint

max_length = 20

trunc_type = 'post'

padding_type = 'post'



# oov_took its set for words out our word index

oov_tok = "<XXX>"

training_size = 6200

seq_len = 12



# based on 80% of the data

training_sentences = train_df.text[0:training_size]

training_labels = train_df.target[0:training_size]



valid_sentences = train_df.text[training_size:]

valid_labels = train_df.target[training_size:]



print('The Shape of training ',training_sentences.shape)

print('The Shape of testing',valid_sentences.shape)
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
# Lets see the first 10 elements

print("THe first word Index are: ")

for x in list(word_index)[0:15]:

    print (" {},  {} ".format(x,  word_index[x]))
training_sequences = tokenizer.texts_to_sequences(training_sentences)

training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(train_df.text[1])

print(training_sequences[1])
# check reverse to see how it works

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Lets see the first 10 elements

print("THe first reverse word Index are: ")

for x in list(reverse_word_index)[0:15]:

    print (" {},  {} ".format(x,  reverse_word_index[x]))
def decode(text):

    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode(training_sequences[1]) # this can be useful in checking predictions
valid_sequences = tokenizer.texts_to_sequences(valid_sentences)

valid_padded = pad_sequences(valid_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# Model Definition with LSTM



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid') # remember this is a binary classification

])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)
start_time = time.time()



num_epochs = 40

history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(valid_padded, valid_labels))



final_time = (time.time()- start_time)/60

print(f'The time in minutes: {final_time}')
model_loss = pd.DataFrame(model.history.history)

model_loss.head()
model_loss[['accuracy','val_accuracy']].plot(ylim=[0,1])
predictions = model.predict_classes(valid_padded)

predictions
from sklearn.metrics import classification_report,confusion_matrix



# Showing Confusion Matrix

def plot_cm(y_true, y_pred, title, figsize=(5,4)):

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    plt.title(title)

    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)

    

# Showing Confusion Matrix

plot_cm(valid_labels,predictions, 'Confution matrix of Tweets', figsize=(7,7))
testing_sequences2 = tokenizer.texts_to_sequences(test_df.text)

testing_padded2 = pad_sequences(testing_sequences2, maxlen=max_length, padding=padding_type, truncating=trunc_type)

predictions = model.predict(testing_padded2)

# sample of submission



sub['target'] = (predictions > 0.5).astype(int)

sub.head()
sub.to_csv("submission.csv", index=False, header=True)
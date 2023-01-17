# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

from gensim.models import Doc2Vec

from sklearn import utils

from sklearn.model_selection import train_test_split

import gensim

from sklearn.linear_model import LogisticRegression

from gensim.models.doc2vec import TaggedDocument

import re

import seaborn as sns

import matplotlib.pyplot as plt

import string

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer
def remove_punctuation(text):

    '''a function for removing punctuation'''

    import string

    # replacing the punctuations with no space, 

    # which in effect deletes the punctuation marks 

    translator = str.maketrans('', '', string.punctuation)

    # return the text stripped of punctuation marks

    return text.translate(translator)





# extracting the stopwords from nltk library

sw = stopwords.words('english')

# displaying the stopwords

np.array(sw);





def stopwords(text):

    '''a function for removing the stopword'''

    # removing the stop words and lowercasing the selected words

    text = [word.lower() for word in text.split() if word.lower() not in sw]

    # joining the list of words with space separator

    return " ".join(text)



stemmer = SnowballStemmer("english")



def stemming(text):    

    '''a function which stems each word in the given text'''

    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text) 

    





def clean_loc(x):

    if x == 'None':

        return 'None'

    elif x == 'Earth' or x =='Worldwide' or x == 'Everywhere':

        return 'World'

    elif 'New York' in x or 'NYC' in x:

        return 'New York'    

    elif 'London' in x:

        return 'London'

    elif 'Mumbai' in x:

        return 'Mumbai'

    elif 'Washington' in x and 'D' in x and 'C' in x:

        return 'Washington DC'

    elif 'San Francisco' in x:

        return 'San Francisco'

    elif 'Los Angeles' in x:

        return 'Los Angeles'

    elif 'Seattle' in x:

        return 'Seattle'

    elif 'Chicago' in x:

        return 'Chicago'

    elif 'Toronto' in x:

        return 'Toronto'

    elif 'Sacramento' in x:

        return 'Sacramento'

    elif 'Atlanta' in x:

        return 'Atlanta'

    elif 'California' in x:

        return 'California'

    elif 'Florida' in x:

        return 'Florida'

    elif 'Texas' in x:

        return 'Texas'

    elif 'United States' in x or 'USA' in x:

        return 'USA'

    elif 'United Kingdom' in x or 'UK' in x or 'Britain' in x:

        return 'UK'

    elif 'Canada' in x:

        return 'Canada'

    elif 'India' in x:

        return 'India'

    elif 'Kenya' in x:

        return 'Kenya'

    elif 'Nigeria' in x:

        return 'Nigeria'

    elif 'Australia' in x:

        return 'Australia'

    elif 'Indonesia' in x:

        return 'Indonesia'

    elif x in top_loc:

        return x

    else: return 'Others'

    

    

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)

new_df = pd.read_csv('../input/nlp-getting-started/train.csv')

final_test = pd.read_csv('../input/nlp-getting-started/test.csv')
new_df['keyword'] = new_df['keyword'].fillna('unknown')

new_df['location'] = new_df['location'].fillna('unknown')





new_df = new_df[['target', 'location', 'text', 'keyword']]

final_test = final_test[['location', 'text', 'keyword']]







new_df['text'] = new_df['text'].apply(remove_punctuation)

new_df['text'] = new_df['text'].apply(stopwords)

new_df['text'] = new_df['text'].apply(stemming)

new_df['text'] = new_df['text'].apply(remove_URL)

new_df['text'] = new_df['text'].apply(remove_html)

new_df['text'] = new_df['text'].apply(remove_emoji)

new_df['text'] = new_df['text'].apply(remove_punct)







final_test['text'] = final_test['text'].apply(remove_punctuation)

final_test['text'] = final_test['text'].apply(stopwords)

final_test['text'] = final_test['text'].apply(stemming)

final_test['text'] = final_test['text'].apply(remove_URL)

final_test['text'] = final_test['text'].apply(remove_html)

final_test['text'] = final_test['text'].apply(remove_emoji)

final_test['text'] = final_test['text'].apply(remove_punct)

new_df.head()
from bs4 import BeautifulSoup



def cleanText(text):

    text = BeautifulSoup(text, "lxml").text

    text = re.sub(r'\|\|\|', r' ', text) 

    text = re.sub(r'http\S+', r'<URL>', text)

    text = text.lower()

    text = text.replace('x', '')

    return text



new_df['text'] = new_df['text'].apply(cleanText)

new_df['keyword'] = new_df['keyword'].apply(cleanText)



final_test['text'] = final_test['text'].apply(cleanText)

final_test['keyword'] = final_test['keyword'].fillna('unknown')

final_test['keyword'] = final_test['keyword'].apply(cleanText)

train, test = train_test_split(new_df, test_size=0.2, random_state=42)

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from collections import Counter



def counter_word (text):

    count = Counter()

    for i in text.values:

        for word in i.split():

            count[word] += 1

    return count
text_values = train["text"]



counter = counter_word(text_values)
# NOTE: comment out if using fasttext

glove_embedding_idx = {}

EMBEDDING_DIM = 100

with open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        glove_embedding_idx[word] = vectors

f.close()
# The maximum number of words to be used. (most frequent)



vocab_size = len(counter)

embedding_dim = 100



# Max number of words in each complaint.

max_length = 20

trunc_type='post'

padding_type='post'



# oov_took its set for words out our word index

oov_tok = "<XXX>"

training_size = 6090

seq_len = 12


training_sentences = new_df.text[0:training_size]

training_labels = new_df.target[0:training_size]



testing_sentences = new_df.text[training_size:]

testing_labels = new_df.target[training_size:]
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
iv = 0

oov = 0



embedding_idx = glove_embedding_idx # swap between embeddings



embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embedding_idx.get(word)

    if embedding_vector is not None:

        iv += 1

        # words not found in the embedding space are all zeros

        embedding_matrix[i] = embedding_vector

    else: oov += 1

        

print('%i tokens in vocab, %i tokens out of vocab' % (iv, oov)) # TODO: must reduce out of vocab
training_sequences = tokenizer.texts_to_sequences(training_sentences)

training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(train.text[1])

print(training_sequences[1])
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode(text):

    return ' '.join([reverse_word_index.get(i, '?') for i in text])
decode(training_sequences[1])
print(train.text[1])

print(training_sequences[1])
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Model Definition with LSTM



model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(max_length,), dtype='int32'),

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length,weights = [embedding_matrix],trainable=False),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

    tf.keras.layers.Dense(14, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')  

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
import time

start_time = time.time()



num_epochs = 10

history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels))



final_time = (time.time()- start_time)/60

print(f'The time in minutos: {final_time}')
model_loss = pd.DataFrame(model.history.history)

model_loss.head()
model_loss[['accuracy','val_accuracy']].plot(ylim=[0,1]);
predictions = model.predict_classes(testing_padded)
testing_sequences2 = tokenizer.texts_to_sequences(final_test.text)

testing_padded2 = pad_sequences(testing_sequences2, maxlen=max_length, padding=padding_type, truncating=trunc_type)
predictions = model.predict(testing_padded2)
sub_sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submit = sub_sample.copy()

submit.target = np.where(predictions > 0.5,1,0)



submit.to_csv('submit_lstm_glove_embeddings_dropout2.csv',index=False)
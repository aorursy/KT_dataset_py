# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

from sklearn.preprocessing import LabelBinarizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from wordcloud import WordCloud,STOPWORDS

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize,sent_tokenize

from bs4 import BeautifulSoup

import re,string,unicodedata

from keras.preprocessing import text, sequence

from nltk.tokenize.toktok import ToktokTokenizer

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

from string import punctuation

from nltk import pos_tag

from nltk.corpus import wordnet

import keras

from keras.models import Sequential

from keras.layers import Dense,Embedding,LSTM,Dropout

import tensorflow as tf
df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines=True)

df.head()
df.isna().sum() # Checking for NaN values
del df['article_link'] # Deleting this column as it is of no use
df.head()
sns.set_style("dark")

sns.countplot(df.is_sarcastic)
stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)
def strip_html(text):

    soup = BeautifulSoup(text, "html.parser")

    return soup.get_text()



#Removing the square brackets

def remove_between_square_brackets(text):

    return re.sub('\[[^]]*\]', '', text)

# Removing URL's

def remove_between_square_brackets(text):

    return re.sub(r'http\S+', '', text)

#Removing the stopwords from text

def remove_stopwords(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            final_text.append(i.strip())

    return " ".join(final_text)

#Removing the noisy text

def denoise_text(text):

    text = strip_html(text)

    text = remove_between_square_brackets(text)

    text = remove_stopwords(text)

    return text

#Apply function on review column

df['headline']=df['headline'].apply(denoise_text)
plt.figure(figsize = (20,20)) # Text that is Not Sarcastic

wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.is_sarcastic == 0].headline))

plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20)) # Text that is Sarcastic

wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.is_sarcastic == 1].headline))

plt.imshow(wc , interpolation = 'bilinear')
words = []

for i in df.headline.values:

    l = []

    for j in i.split():

        l.extend(i.split())

        break

    words.append(l)    
import gensim

#Dimension of vectors we are generating

EMBEDDING_DIM = 100



#Creating Word Vectors by Word2Vec Method (takes time...)

w2v_model = gensim.models.Word2Vec(sentences = words , size=EMBEDDING_DIM , window = 5 , min_count = 1)
#vocab size

len(w2v_model.wv.vocab)

#We have now represented each of 38071 words by a 100dim vector.
# For determining size of input...



# Making histogram for no of words in news shows that all news article are under 20 words.

# Lets keep each news small and truncate all news to 20 while tokenizing

plt.hist([len(j) for j in words], bins = 100)

plt.show()
tokenizer = text.Tokenizer(num_words=35000)

tokenizer.fit_on_texts(words)

tokenized_train = tokenizer.texts_to_sequences(words)

x = sequence.pad_sequences(tokenized_train, maxlen = 20)
# Adding 1 because of reserved 0 index

# Embedding Layer creates one more vector for "UNKNOWN" words, or padded words (0s). This Vector is filled with zeros.

# Thus our vocab size inceeases by 1

vocab_size = len(tokenizer.word_index) + 1
# Function to create weight matrix from word2vec gensim model

def get_weight_matrix(model, vocab):

    # total vocabulary size plus 0 for unknown words

    vocab_size = len(vocab) + 1

    # define weight matrix dimensions with all 0

    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    # step vocab, store vectors using the Tokenizer's integer mapping

    for word, i in vocab.items():

        weight_matrix[i] = model[word]

    return weight_matrix
#Getting embedding vectors from word2vec and usings it as weights of non-trainable keras embedding layer

embedding_vectors = get_weight_matrix(w2v_model, tokenizer.word_index)
#Defining Neural Network

model = Sequential()

#Non-trainable embeddidng layer

model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=20, trainable=True))

#LSTM 

model.add(LSTM(units=128))

model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])



del embedding_vectors
model.summary()
x_train, x_test, y_train, y_test = train_test_split(x, df.is_sarcastic , test_size = 0.3 , random_state = 0) 
history = model.fit(x_train, y_train, batch_size = 128 , validation_data = (x_test,y_test) , epochs = 5)
print("Accuracy of the model on Training Data is - " , model.evaluate(x_train,y_train)[1]*100)

print("Accuracy of the model on Testing Data is - " , model.evaluate(x_test,y_test)[1]*100)
epochs = [i for i in range(5)]

fig , ax = plt.subplots(1,2)

train_acc = history.history['acc']

train_loss = history.history['loss']

val_acc = history.history['val_acc']

val_loss = history.history['val_loss']

fig.set_size_inches(20,10)



ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')

ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')

ax[0].set_title('Training & Testing Accuracy')

ax[0].legend()

ax[0].set_xlabel("Epochs")

ax[0].set_ylabel("Accuracy")



ax[1].plot(epochs , train_loss , 'go-' , label = 'Training Loss')

ax[1].plot(epochs , val_loss , 'ro-' , label = 'Testing Loss')

ax[1].set_title('Training & Testing Loss')

ax[1].legend()

ax[1].set_xlabel("Epochs")

ax[1].set_ylabel("Loss")

plt.show()
pred = model.predict_classes(x_test)

pred[:5]
cm = confusion_matrix(y_test,pred)

cm
cm = pd.DataFrame(cm , index = ['Not Sarcastic','Sarcastic'] , columns = ['Not Sarcastic','Sarcastic'])

plt.figure(figsize = (10,10))

sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['Not Sarcastic','Sarcastic'] , yticklabels = ['Not Sarcastic','Sarcastic'])
x_train,x_test,y_train,y_test = train_test_split(df.headline,df.is_sarcastic, test_size = 0.3 , random_state = 0)
max_features = 10000

maxlen = 200
tokenizer = text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(x_train)

tokenized_train = tokenizer.texts_to_sequences(x_train)

x_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
tokenized_test = tokenizer.texts_to_sequences(x_test)

X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
EMBEDDING_FILE = '../input/glove-twitter/glove.twitter.27B.200d.txt'
def get_coefs(word, *arr): 

    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

#change below line if computing normal stats is too slow

embedding_matrix = embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
batch_size = 128

epochs = 3

embed_size = 200
#Defining Neural Network

model = Sequential()

#Non-trainable embeddidng layer

model.add(Embedding(max_features, output_dim=embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=True))

#LSTM 

model.add(LSTM(units=128))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, batch_size = batch_size , validation_data = (X_test,y_test) , epochs = epochs)
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100)
epochs = [i for i in range(3)]

fig , ax = plt.subplots(1,2)

train_acc = history.history['accuracy']

train_loss = history.history['loss']

val_acc = history.history['val_accuracy']

val_loss = history.history['val_loss']

fig.set_size_inches(20,10)



ax[0].plot(epochs , train_acc , 'g' , label = 'Training Accuracy')

ax[0].plot(epochs , val_acc , 'r' , label = 'Testing Accuracy')

ax[0].set_title('Training & Testing Accuracy')

ax[0].legend()

ax[0].set_xlabel("Epochs")

ax[0].set_ylabel("Accuracy")



ax[1].plot(epochs , train_loss , 'g' , label = 'Training Loss')

ax[1].plot(epochs , val_loss , 'r' , label = 'Testing Loss')

ax[1].set_title('Training & Testing Loss')

ax[1].legend()

ax[1].set_xlabel("Epochs")

ax[1].set_ylabel("Loss")

plt.show()
pred = model.predict_classes(X_test)

pred[:5]
print(classification_report(y_test, pred, target_names = ['Not Sarcastic','Sarcastic']))
cm = confusion_matrix(y_test,pred)

cm
cm = pd.DataFrame(cm , index = ['Not Sarcastic','Sarcastic'] , columns = ['Not Sarcastic','Sarcastic'])

plt.figure(figsize = (10,10))

sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['Not Sarcastic','Sarcastic'] , yticklabels = ['Not Sarcastic','Sarcastic'])
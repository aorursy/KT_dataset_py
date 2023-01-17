# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)







from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils.vis_utils import plot_model

from keras.models import Model, Sequential

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import Embedding

from keras.layers import LSTM, GRU, RNN, Recurrent, SimpleRNN

from keras.layers import Bidirectional

from keras.layers.convolutional import Conv1D 

from keras.layers.convolutional import MaxPooling1D

from keras.layers.merge import concatenate





from string import punctuation

from nltk.corpus import stopwords

from pickle import load, dump



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
print(os.listdir("../input/sentiment-analysis-on-movie-reviews"))
train=pd.read_csv("../input/sentiment-analysis-on-movie-reviews/train.tsv", sep="\t")

test=pd.read_csv("../input/sentiment-analysis-on-movie-reviews/test.tsv", sep="\t")
train.head()
import string

punctuations = string.punctuation



def punct_remover(my_str):

    my_str = my_str.lower()

    no_punct = ""

    for char in my_str:

       if char not in punctuations:

           no_punct = no_punct + char

    return no_punct



punctuations
def clean_doc(doc):

    tokens=doc.split()

    table=str.maketrans('','',punctuation)

    tokens=[w.translate(table) for w in tokens]

    tokens=[word for word in tokens if word.isalpha()]

    stop_words= set(stopwords.words('english'))

    

    tokens=[word for word in tokens if not word in stop_words]

    

    tokens=[word for word in tokens if len(word)>1]

    tokens=' '.join(tokens)

    return tokens
def save_dataset(dataset, filename):

    dump(dataset, open(filename, 'wb'))

    print('Saved: %s' % filename)
train_documents = list()

#vocab=Counter()

for i in range(len(train["Phrase"])):

    tokens = clean_doc(train['Phrase'][i])

    train_documents.append(tokens)
test_documents = list()

#vocab=Counter()

for i in range(len(test["Phrase"])):

    tokens = clean_doc(test['Phrase'][i])

    test_documents.append(tokens)
trainX=train_documents

trainy=train["Sentiment"].tolist()

testX=test_documents

print(len(trainX))

print(len(trainy))

print (len(testX))
save_dataset([trainX,trainy],'training.pkl')
def load_dataset(filename):

    return load(open(filename, 'rb'))
def create_tokenizer(lines):

    tokenizer= Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer
def max_length(lines):

    return max([len(s.split()) for s in lines])
def encode_text(tokenizer, lines, length):

    encoded=tokenizer.texts_to_sequences(lines)

    padded = pad_sequences(encoded, maxlen=length, padding='post')

    return padded
print(os.listdir("../input"))
embeddings_index = dict()

#path1 = "../Untitled⁩/Users⁩⁨/tsegayemisikir⁩/Desktop⁩/Coursera⁩/AEE_DL⁩/glove.6B⁩/glove.6B.100d.txt"

f = open('../input/word2vec1/glove.6B.300d.txt' ,encoding='utf8')

for line in f:

    values = line.split(" ")

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



embed_token = create_tokenizer(trainX)

vocabulary_size = 90000



embedding_matrix = np.zeros((vocabulary_size, 300))

for word, index in embed_token.word_index.items():

    if index > vocabulary_size - 1:

        break

    else:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[index] = embedding_vector
def define_model(length, vocab_size):

    # channel 1

    inputs1 = Input(shape=(length,))

    embedding1 = Embedding(vocabulary_size, 300, weights=[embedding_matrix])(inputs1)

    conv1 = Conv1D(filters=32, kernel_size=2, activation='relu')(embedding1)

    drop1 = Dropout(0.5)(conv1)

    pool1 = MaxPooling1D(pool_size=2)(conv1)

    gru1 = Bidirectional(GRU(32, return_sequences = True))(pool1)

    flat1 = Flatten()(gru1)

   

    dense1 = Dense(32, activation='relu')(flat1)

    outputs = Dense(num_classes, activation='softmax')(dense1)

    model = Model(inputs= inputs1, outputs=outputs)

    # compile

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # summarize

    print(model.summary())

    plot_model(model, show_shapes=True, to_file='multichannel.png')

    return model
trainLines, trainLabels = load_dataset('training.pkl')
# create tokenizer

token_train = create_tokenizer(trainLines)

token_test = create_tokenizer(testX)

# calculate max document length

length = max_length(trainLines)

# calculate vocabulary size
# calculate vocabulary size

vocab_size = len(token_train.word_index) + 1

print('Max document length: %d' % length)

print('Vocabulary size: %d' % vocab_size)
trainX = encode_text(token_train, trainLines, length)

testX= encode_text(token_test, testX,length)

print(trainX.shape)

print(testX.shape)
from keras.utils import to_categorical

y_train=to_categorical(trainLabels)

num_classes=y_train.shape[1]
X_train=np.array(trainX)

X_test=np.array(testX)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.3, random_state=42)
model = define_model(length, vocab_size)
history= model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=9)
prediction= model.predict(X_test)
submission=pd.read_csv("../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv", sep=",")

submission.Sentiment=np.argmax(prediction,axis=1)

submission.to_csv('Submission.csv',index=False)
submission
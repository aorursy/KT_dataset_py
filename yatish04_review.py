!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Gift_Cards.json.gz
!gzip -d Gift_Cards.json.gz
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import json
import string
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
import os

nltk.download('punkt')

vocab={}
rev_vocab={}
WINDOW_SIZE = 3
stopwords=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def preprocess(sentence_list,ratings):
    new_ratings=[]
    sentence_list_process=[]
    for i in range(len(sentence_list)):
        try:
            text = sentence_list[i]
            text = text.lower()
            text= text.translate(str.maketrans('','',string.punctuation))
            tokens = word_tokenize(text)

            words = [word for word in tokens if word.isalpha()]
            words = [w for w in words if not w in stopwords]
            
            if len(words)<=2:
                continue
            sentence_list_process.append(words)
            new_ratings.append(ratings[i])
        except:
            pass
    return sentence_list_process, new_ratings


with open('Gift_Cards.json') as f:
    data=f.readlines()

for i in range(len(data)):
    data[i] = json.loads(data[i])

sentences=[]
ratings=[]


for review in data:
    try:
        sent = review["reviewText"]
        sentences.append(sent)
        ratings.append(int(review["overall"]))
    except:
        pass
processed_sentences,new_ratings = preprocess(sentences,ratings)
print("preprocess done")

def build_vocab(sentences):
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                idx = len(vocab)+1
                vocab[word] = idx
                rev_vocab[idx] = word

build_vocab(processed_sentences)
print("build vocab done")



def to_one_hot(word, vocab_size):
    temp = np.zeros(vocab_size)
    temp[vocab[word]-1] = 1
    return temp

# print(X,Y)
lens={}
for sent in processed_sentences:
    if len(sent) in lens:
        lens[len(sent)]+=1
    else:
        lens[len(sent)]=1
        
mean=0
var =0
for i in lens:
    mean+=lens[i]*i
sums=mean
print(mean/len(processed_sentences))
for i in processed_sentences:
    var+=((len(i)-mean))**2
import math
print(math.sqrt(var/len(processed_sentences)))
# processed_sentences
!wget nlp.stanford.edu/data/glove.6B.zip
! unzip glove.6B.zip


embeddings_index = {}
f = open(os.path.join(".", 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# weights is #vocab * 100
weights = np.random.rand(len(vocab)+1,100)
c=0
for word in vocab:
    if word in embeddings_index:
        row = embeddings_index[word]
        c+=1
        weights[vocab[word],:] = row
        

embed_size = 100
lstm_out = 200
batch_size = 32
vocab_size = len(vocab)
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()
model.add(Embedding(input_dim=vocab_size+1, output_dim=embed_size,weights=[weights]))
model.add(LSTM(lstm_out,dropout=0.2))
model.add(Dense(5,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
Y=[]
for ratings in new_ratings:
    temp = np.zeros(5)
    temp[ratings-1] = 1
    Y.append(temp)    
Y=np.array(Y)
processed_sentences[0]
lens
X_coded =[]
for sent in processed_sentences:
    tx=[]
    
    for word in sent:
        tx.append(vocab[word])
    X_coded.append(tx)

from keras.preprocessing.sequence import pad_sequences
X=np.array(pad_sequences(X_coded, maxlen=100))
print(Y.shape,X.shape)
j=0
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 36)
def generator(c):
    global j
    sz=len(X_train)
    while True:
        x=[]
        y=[]
        for t in range(c):
            x.append(X_train[j])
            y.append(Y_train[j])
            j=j+1
            j%=sz
        yield np.array(x),np.array(y)
history = model.fit_generator(generator(32),steps_per_epoch=4,epochs=4000)
score,acc=model.evaluate(X_valid,Y_valid,verbose=2,batch_size=batch_size)
print(score,acc)

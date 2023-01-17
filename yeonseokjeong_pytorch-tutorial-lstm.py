import torch

import torch.nn as nn

import torch.nn.functional as F

import pandas as pd

from nltk.tokenize import word_tokenize, sent_tokenize

import matplotlib.pyplot as plt

%matplotlib inline

from tensorflow.keras.preprocessing.text import Tokenizer

from torchtext import data

from torchtext.data import TabularDataset

from tqdm.notebook import tqdm
import torch

import torchtext

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import random
SEED = 5

random.seed(SEED)

torch.manual_seed(SEED)
# dataset

TRAIN_SIZE = 0.7

# model

BATCH_SIZE = 64

LEARNING_RATE = 0.001

EPOCHS = 30

DROP_RATE = 0.3
USE_CUDA = torch.cuda.is_available()

DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)
### 1> set several paths

PATH_TRAIN = '../input/nlp-getting-started/train.csv'

PATH_TEST = '../input/nlp-getting-started/test.csv'



### 2> read_csv

df_train = pd.read_csv(PATH_TRAIN)

df_test = pd.read_csv(PATH_TEST)
print(df_train.shape)

print(df_test.shape)
df = df_train.append(df_test, sort=False)

df.shape
import re

import string
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    

    return url.sub('', text)



def remove_html(text):

    html = re.compile(r'<.*?>')

    

    return html.sub('', text)

    

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

    table = str.maketrans('', '', string.punctuation)

    

    return text.translate(table)
df['text'] = df['text'].apply(lambda x: remove_URL(x))

df['text'] = df['text'].apply(lambda x: remove_html(x))

df['text'] = df['text'].apply(lambda x: remove_emoji(x))

df['text'] = df['text'].apply(lambda x: remove_punct(x))
from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

import nltk

nltk.download('wordnet')
keywords = df_train.keyword.unique()[1:]

keywords = list(map(lambda x: x.replace('%20', ' '), keywords))



wnl = WordNetLemmatizer()



def lemmatize_sentence(sentence):

    sentence_words = sentence.split(' ')

    new_sentence_words = list()

    

    for sentence_word in sentence_words:

        sentence_word = sentence_word.replace('#', '')

        new_sentence_word = wnl.lemmatize(sentence_word.lower(), wordnet.VERB)

        new_sentence_words.append(new_sentence_word)

        

    new_sentence = ' '.join(new_sentence_words)

    new_sentence = new_sentence.strip()

    

    return new_sentence
df['text'] = df['text'].apply(lambda x: lemmatize_sentence(x))
df_train = df.iloc[:len(df_train)]

df_test = df.iloc[len(df_train):]
df_train = df_train[['id','text','target']]

df_test = df_test[['id','text']]
import os
if not os.path.exists('preprocessed_train.csv'):

    df_train.to_csv('preprocessed_train.csv', index = False)

    

if not os.path.exists('preprocessed_test.csv'):

    df_test.to_csv('preprocessed_test.csv', index = False)
TEXT = torchtext.data.Field(sequential=True, 

                            tokenize='spacy', 

                            lower=True, 

                            include_lengths=True, 

                            batch_first=True, 

                            fix_length=25)

LABEL = torchtext.data.Field(use_vocab=True,

                           sequential=False,

                           dtype=torch.float16)

ID = torchtext.data.Field(use_vocab=False,

                         sequential=False,

                         dtype=torch.float16)
from torchtext.data import TabularDataset
trainset = TabularDataset(path='preprocessed_train.csv', format='csv', skip_header=True,

                            fields=[('id', ID), ('text', TEXT), ('target', LABEL)])

testset = TabularDataset(path='preprocessed_test.csv', format='csv', skip_header=True,

                            fields=[('id', ID), ('text', TEXT)])
from torchtext.vocab import Vectors, GloVe
TEXT.build_vocab(trainset, testset, 

                 max_size=20000, min_freq=10,

                 vectors=GloVe(name='6B', dim=300))  # We use it for getting vocabulary of words

LABEL.build_vocab(trainset)

ID.build_vocab(trainset, testset)
trainset, valset = trainset.split(split_ratio = TRAIN_SIZE, random_state=random.getstate(),

                                  strata_field = 'target', stratified=True)
train_iter = torchtext.data.Iterator(dataset = trainset, batch_size = BATCH_SIZE, device = DEVICE,

                                     train=True, shuffle=True, repeat=False, sort = False)

val_iter = torchtext.data.Iterator(dataset = valset, batch_size = BATCH_SIZE, device = DEVICE,

                                  train=True, shuffle=True, repeat=False)

test_iter = torchtext.data.Iterator(dataset = testset, batch_size = BATCH_SIZE, device = DEVICE,

                                   train=False, shuffle=False, repeat=False)
word_embeddings = TEXT.vocab.vectors

vocab_size = len(TEXT.vocab)

n_classes = 2
class LSTM_model(nn.Module):

    def __init__(self, n_layers, hidden_dim, n_vocab, embedding_dim, n_classes, dropout_p = DROP_RATE):

        super(LSTM_model, self).__init__()

        self.n_layers = n_layers

        self.embed = nn.Embedding(n_vocab, embedding_dim)

        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(dropout_p)

        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True)

        self.out = nn.Linear(self.hidden_dim, n_classes)

        

    def forward(self, x):

        # x = [64, 27]

        x = self.embed(x)

        # x = [64, 27, 128]

        h_0 = self._init_state(batch_size = x.size(0))#첫 번째 은닉 벡터 정의

        # h_0 = [1, 64, 256]

        x, _ = self.lstm(x,(h_0,h_0))

        # x = [64, 27, 256]

        h_t = x[:,-1,:]

        # h_t = [64, 256]

        self.dropout(h_t)

        logit = self.out(h_t)

        # logit = [64, 2]

        return logit

    

    def _init_state(self, batch_size = 1):

        weight = next(self.parameters()).data

        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
def train(model, optimizer, train_iter):

    model.train()

    acc, total_loss = 0, 0

    for b,batch in enumerate(train_iter):

        x, y = batch.text[0], batch.target

        y.sub_(1)

        y = y.type(torch.LongTensor)

        x = x.to(DEVICE)

        y = y.data.to(DEVICE)

        optimizer.zero_grad()# 기울기 0으로 초기화

        logit = model(x)

        loss = F.cross_entropy(logit, y, reduction = 'mean')

        total_loss += loss.item()

        acc += (logit.max(1)[1].view(y.size()).data == y.data).sum()

        loss.backward()

        optimizer.step()

    size = len(train_iter.dataset)

    avg_loss = total_loss / size

    avg_accuracy = 100. * acc / size

    return avg_loss, avg_accuracy
def evaluate(model, val_iter):

    model.eval()

    acc, total_loss = 0., 0.

    for batch in val_iter:

        x, y = batch.text[0], batch.target

        y.sub_(1)

        y = y.type(torch.LongTensor)

        x = x.to(DEVICE)

        y = y.data.to(DEVICE)

        logit = model(x)

        loss = F.cross_entropy(logit, y, reduction = 'sum')#오차의 합 구하고 total_loss에 더해줌

        total_loss += loss.item()

        acc += (logit.max(1)[1].view(y.size()).data == y.data).sum()

    size = len(val_iter.dataset)

    avg_loss = total_loss / size

    avg_accuracy = 100. * acc / size

    return avg_loss, avg_accuracy
model = LSTM_model(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
best_val_loss = None

for e in tqdm(range(1, EPOCHS + 1)):

    train_loss, train_accuracy = train(model, optimizer, train_iter)

    val_loss, val_accuracy = evaluate(model, val_iter)



    print("<<e : %d>> <<train_loss : %f>> <<train_accuracy : %f>> <<val_loss : %f>> <<val_accuracy : %f>>"%(e, train_loss, train_accuracy, val_loss, val_accuracy))
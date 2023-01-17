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

import torchtext

import random

from torchtext.data import TabularDataset

import numpy as np

import os

from torchtext import data

import torch.nn as nn

import torch

import torch.optim as optim

import time

from sklearn.metrics import roc_auc_score,accuracy_score

import spacy

import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
# hyperparams

TEXT_LENGTH = 100

EMBEDDING_SIZE = 50

BATCH_SIZE = 64

VOCAB_SIZE=20000
filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

nlp = spacy.load("en")

def tokenizer(text):

    filtered = ''.join([c if c not in filters else '' for c in text])

    return [token.text for token in nlp.tokenizer(filtered) if not token.is_space]
TEXT = data.Field(lower=True, batch_first=True,fix_length=TEXT_LENGTH, preprocessing=None, tokenize=tokenizer)

LABEL = data.Field(sequential=False,is_target=True, use_vocab=False, pad_token=None, unk_token=None)



datafields = [('id', None),

              ('comment_text', TEXT), 

              ("toxic", LABEL), 

              ("severe_toxic", LABEL),

              ('obscene', LABEL), 

              ('threat', LABEL),

              ('insult', LABEL),

              ('identity_hate', LABEL)]





alldata = TabularDataset(

    path='/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv',

    format='csv',

    skip_header=True,

    fields=datafields,)
random.seed(17)

train,dev = alldata.split(split_ratio=0.9, random_state=random.getstate())
TEXT.build_vocab(train, max_size=20000, min_freq=5)
random.seed(1234)

train_iterator, valid_iterator = data.BucketIterator.splits((train, dev),

                                                            batch_size=BATCH_SIZE,

                                                            device=device,

                                                            shuffle=True,

                                                            sort_key=lambda x: len(x.comment_text))
class NNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx, embeddings, text_length, lstm_hidden_size):

        super().__init__()

        

        #self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=pad_idx)

        self.embeddings = nn.Embedding(vocab_size,embedding_dim,pad_idx)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)

        self.max_pool = nn.MaxPool2d((text_length,1))

        self.fc1 = nn.Linear(lstm_hidden_size, 50)

        self.fc2 = nn.Linear(50, output_dim)



    def forward(self, text):

        a1 = self.embeddings(text)

        a2 = self.lstm(a1)[0]

        a3 = self.max_pool(a2).squeeze(1)

        a4 = F.relu(self.fc1(a3))

        a5 = self.fc2(a4)

        return a5
OUTPUT_DIM = 6

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = NNet(len(TEXT.vocab), EMBEDDING_SIZE, OUTPUT_DIM, PAD_IDX, TEXT.vocab.vectors,TEXT_LENGTH, 150).to(device)



def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
def fit_epoch(iterator, model, optimizer, criterion):

    train_loss = 0

    train_acc = 0

    model.train()

    all_y = []

    all_y_hat = []

    for batch in iterator:

        optimizer.zero_grad()

        y = torch.stack([batch.toxic,

                         batch.severe_toxic,

                         batch.obscene,

                         batch.threat,

                         batch.insult,

                         batch.identity_hate],dim=1).float().to(device)

        y_hat = model(batch.comment_text.to(device))

        loss = criterion(y_hat, y)

        train_loss += loss.item()

        loss.backward()

        optimizer.step()

        all_y.append(y)

        all_y_hat.append(y_hat)

    y = torch.cat(all_y,dim=0)

    y_hat = torch.cat(all_y_hat,dim=0)

    roc = roc_auc_score(y.cpu(),y_hat.sigmoid().detach().cpu())

    return train_loss / len(iterator.dataset), roc



def test_epoch(iterator, model, criterion):

    train_loss = 0

    train_acc = 0

    model.eval()

    all_y = []

    all_y_hat = []

    for batch in iterator:

        y = torch.stack([batch.toxic,

                         batch.severe_toxic,

                         batch.obscene,

                         batch.threat,

                         batch.insult,

                         batch.identity_hate],dim=1).float().to(device)

        with torch.no_grad():

            y_hat = model(batch.comment_text.to(device))

        loss = criterion(y_hat, y)

        train_loss += loss.item()

        all_y.append(y)

        all_y_hat.append(y_hat)

    y = torch.cat(all_y,dim=0)

    y_hat = torch.cat(all_y_hat,dim=0)

    roc = roc_auc_score(y.cpu(),y_hat.sigmoid().detach().cpu())

    return train_loss / len(iterator.dataset), roc
def train_n_epochs(n, lr, wd):



    criterion = nn.BCEWithLogitsLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(n):

        start_time = time.time()

        train_loss, train_roc = fit_epoch(train_iterator, model, optimizer, criterion)

        valid_loss, valid_roc = test_epoch(valid_iterator, model, criterion)



        secs = int(time.time() - start_time)

        mins = secs / 60

        secs = secs % 60



        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))

        print(f'\tLoss: {train_loss:.4f}(train)\t|\troc: {train_roc :.6f} (train)')

        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\troc: {valid_roc:.6f} (valid)') 
#train_n_epochs(3,0.01,0)
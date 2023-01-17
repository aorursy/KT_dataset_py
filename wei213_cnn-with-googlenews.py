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
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



max_features = 100000



tokenizer = Tokenizer(num_words=max_features, lower=True)

full_text = list(train_df['question_text'].values)

tokenizer.fit_on_texts(full_text)



train_tokenized = tokenizer.texts_to_sequences(train_df['question_text'].fillna('missing'))

test_tokenized = tokenizer.texts_to_sequences(test_df['question_text'].fillna('missing'))
max_len = 72

train_X = pad_sequences(train_tokenized, maxlen=max_len)

test_X = pad_sequences(test_tokenized, maxlen=max_len)

train_y = train_df.target.values
from gensim.models import KeyedVectors

from gensim.test.utils import datapath, get_tmpfile

from gensim.scripts.glove2word2vec import glove2word2vec



news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

embeddings = KeyedVectors.load_word2vec_format(news_path, binary=True)

mean = np.mean(embeddings.vectors)

std = np.std(embeddings.vectors)

_, size = embeddings.vectors.shape

num_words = min(len(tokenizer.word_index), max_features)

embedding_matrix = np.random.normal(mean, std, (num_words, size))

for word, index in tokenizer.word_index.items():

    if index >= max_features: continue

    try:

        word_vector = embeddings.get_vector(word)

        embedding_matrix[index] = word_vector

    except:

        pass

print(embedding_matrix.shape)
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import random

from tqdm import  tqdm

tqdm.pandas()



def seed_torch(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

seed = 0

seed_torch(seed)



def sigmoid(x):

    return 1 / (1 + np.exp(-x))



class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(max_features, size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False

        

        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, size))

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(4, size))

        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(5, size))

        

        self.lstm = nn.LSTM(100, 50)

        self.fc = nn.Linear(50, 1)

        self.dropout = nn.Dropout(0.1)

        

    def forward(self, text):

        embedded = self.embedding(text)

        embedded = embedded.unsqueeze(1)

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))

        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))

        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        # print('conved_0', conved_0.shape)

        # print('conved_1', conved_1.shape)

        # print('conved_2', conved_2.shape)

        pooled_0 = F.adaptive_avg_pool1d(conved_0, 50).squeeze(2)

        pooled_1 = F.adaptive_avg_pool1d(conved_1, 50).squeeze(2)

        pooled_2 = F.adaptive_avg_pool1d(conved_2, 50).squeeze(2)

        # print('pooled_0', pooled_0.shape)

        # print('pooled_1', pooled_1.shape)

        # print('pooled_2', pooled_2.shape)

        cat = torch.cat((pooled_0, pooled_1, pooled_2), dim=2)

        # print('cat',cat.shape)

        cat = self.dropout(cat)

        cat = cat.transpose(1,2).transpose(0,1)

        # print('cat', cat.shape)

        output, (hn, cn) = self.lstm(cat)

        

        return self.fc(hn.squeeze(0))

        

        

    
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import DataLoader, TensorDataset

import time

import re

import random

from sklearn.metrics import f1_score



splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=0).split(train_X, train_y))

train_epochs = 5

batch_size = 64



print(type(train_X), train_X.shape)

print(type(train_y), train_y.shape)

train_X = torch.Tensor(train_X)

train_y = torch.Tensor(train_y)



train_preds = np.zeros((len(train_X)))

test_preds = np.zeros((len(test_X)))



x_test_cuda = torch.tensor(test_X, dtype=torch.long).cuda()

test = torch.utils.data.TensorDataset(x_test_cuda)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)



for i, (train_idx, valid_idx) in enumerate(splits):

    print('Fold:{0}'.format(i))

    x_train_fold = torch.tensor(train_X[train_idx], dtype=torch.long).cuda()

    y_train_fold = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).cuda()

    x_val_fold = torch.tensor(train_X[valid_idx], dtype=torch.long).cuda()

    y_val_fold = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).cuda()

    

    train_data = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)

    valid_data = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    

    model = CNN()

    model.cuda()

    loss_function = nn.BCEWithLogitsLoss()

    optimiser = optim.Adam(model.parameters())

    

    for epoch in range(train_epochs):

        start_time = time.time()

        model.train()

        avg_loss = 0.0

        # train  model

        for x_batch, y_batch in tqdm(train_loader, disable=True):

            y_predicted = model(x_batch)

            loss = loss_function(y_predicted, y_batch)

            optimiser.zero_grad()

            loss.backward()

            optimiser.step()

            avg_loss += loss.item()/len(train_loader)



        # evaluate model

        model.eval()

        val_predicted_fold = np.zeros(x_val_fold.size(0))

        test_predicted_fold = np.zeros(len(test_X))

        avg_val_loss =0.0



        for i, (x_batch, y_batch) in enumerate(valid_loader):

            y_predicted = model(x_batch).detach()

            avg_val_loss += loss_function(y_predicted, y_batch).item()/len(valid_loader)

            val_predicted_fold[i*batch_size:(i+1)*batch_size] = sigmoid(y_predicted.cpu().numpy())[:,0]   



        elapsed_time = time.time() - start_time

        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1,

                                                        train_epochs, avg_loss, avg_val_loss, elapsed_time))

    # predict testing dataset

    test_predicted_fold = np.zeros(len(test_X))

    for i, (x_batch,) in enumerate(test_loader):

        y_pred = model(x_batch).detach()

        test_predicted_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    

    train_preds[valid_idx] = val_predicted_fold

    test_preds += test_predicted_fold / len(splits) # take average
def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in tqdm([i * 0.01 for i in range(100)]):

        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result
search_result = threshold_search(train_y, train_preds)

search_result
sub = pd.read_csv('../input/sample_submission.csv')

sub.prediction = test_preds > search_result['threshold']

sub.to_csv("submission.csv", index=False)
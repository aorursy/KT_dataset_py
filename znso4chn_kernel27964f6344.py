import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)



TRAIN_PATH = "/kaggle/input/nlp-getting-started/train.csv"



EMBED_DIM = 200

NUM_CLASS = 2

CACHE_PATH = '/kaggle/working/vector_cache/'

VECTOR_FILE = '/kaggle/input/glove-twitter/glove.twitter.27B.200d.txt'

import os

if not os.path.exists(CACHE_PATH):

   os.mkdir(CACHE_PATH)
from torchtext import data

from torchtext.data.utils import get_tokenizer



import re

re_url = re.compile("(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]")

def my_filter(str):

    if str[0] == '#':

        return str[1:]

    elif re_url.match(str):

        return 'url'

    else: return str



tokenizer = get_tokenizer('spacy')

text_field = data.Field(tokenize=tokenizer, lower=True, preprocessing=lambda x:[my_filter(s) for s in x])

label_field = data.Field(sequential=False, use_vocab=False, is_target=True)

fields = [('id', None), ('keyword', None), ('location', None), ('text', text_field), ('target', label_field)]
train_dataset, valid_dataset = data.TabularDataset(TRAIN_PATH, 'csv', fields, skip_header= True).split(0.9)



from torchtext.vocab import Vectors

vector = Vectors(name=VECTOR_FILE, cache=CACHE_PATH)

text_field.build_vocab(train_dataset, vectors=vector)

EMBED_W = text_field.vocab.vectors

VOCAB_SIZE = len(text_field.vocab)



BATCH_SIZE = 8 #@param {type:"integer"}

train_iter, valid_iter = data.BucketIterator.splits(

    (train_dataset, valid_dataset), (BATCH_SIZE, BATCH_SIZE), device=device, sort=True, sort_key=lambda x: len(x.text))
import torch.nn as nn

import torch.nn.functional as F



# Definition of network

class TextClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):

        super(TextClassifier, self).__init__()

        self._lstm_hidden_size = 100

        self.embed = nn.Embedding.from_pretrained(EMBED_W)

        self.gru = nn.GRU(input_size=embed_dim, hidden_size=self._lstm_hidden_size, num_layers=1, bidirectional=True)

        self.ln = nn.LayerNorm(2*self._lstm_hidden_size)

        self.fc = nn.Linear(2*self._lstm_hidden_size, 20)

        self.drop = nn.Dropout(0.3)

        self.decoder = nn.Linear(20, num_class)



    def forward(self, text):

        embedded = self.embed(text)

        gru_out, gru_hidden = self.gru(embedded)

        x = gru_out[-1]

        x = self.ln(x)

        x = self.drop(x)

        x = F.relu(self.fc(x))

        x = self.decoder(x)

        return F.softmax(x, dim=1)



model = TextClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4,0.6])).to(device)
def train_func(it):

    # Train the model

    train_loss = 0

    acc = 0

    l1 = torch.zeros(1).to(device)

    confusion_matrix = torch.zeros((NUM_CLASS, NUM_CLASS)).to(device)

    for epoch, batch in enumerate(it):

        optimizer.zero_grad()

        output = model(batch.text)

        loss = criterion(output, batch.target)

        train_loss += loss.item()

        loss.backward()

        optimizer.step()

        acc += (output.argmax(1) == batch.target).sum().item()

        for i_predict, i_actual in zip(output.argmax(1), batch.target):

            confusion_matrix[i_predict, i_actual] += 1

    return train_loss / len(it.dataset), acc / len(it.dataset), confusion_matrix



def test(it):

    test_loss = 0

    acc = 0

    confusion_matrix = torch.zeros((NUM_CLASS, NUM_CLASS)).to(device)

    for epoch, batch in enumerate(it):

        with torch.no_grad():

            output = model(batch.text)

            loss = criterion(output, batch.target)

            test_loss += loss.item()

            acc += (output.argmax(1) == batch.target).sum().item()

            for i_predict, i_actual in zip(output.argmax(1), batch.target):

                        confusion_matrix[i_predict, i_actual] += 1

    return test_loss / len(it.dataset), acc / len(it.dataset), confusion_matrix



def f1_score(m):

    ret = []

    classes = set(range(NUM_CLASS))

    for i in range(NUM_CLASS):

        js = list(classes.difference([i]))

        TP = m[i, i]

        FP = sum(m[i, js])

        FN = sum(m[js, i])

        ret.append(2*TP/(2*TP+FP+FN))

    return ret
import time

train_losses = []

valid_losses = []

train_accuracy = []

valid_accuracy = []

train_cm = 0

valid_cm = 0



N_EPOCHS = 40

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc, train_cm = train_func(train_iter)

    valid_loss, valid_acc, valid_cm = test(valid_iter)

    train_losses.append(train_loss)

    valid_losses.append(valid_loss)

    train_accuracy.append(train_acc)

    valid_accuracy.append(valid_acc)

    

    secs = int(time.time() - start_time)

    mins = secs / 60

    secs = secs % 60



    print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))

    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')

    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
import matplotlib.pyplot as plt

import numpy as np

learning_curve = plt.figure()

plt.subplot(211)

plt.plot(np.arange(1, N_EPOCHS+1), train_losses, 'r--', np.arange(1, N_EPOCHS+1), valid_losses, 'b' )

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.yscale('logit')

plt.subplot(212)

plt.plot(np.arange(1, N_EPOCHS+1), train_accuracy, 'r--', np.arange(1, N_EPOCHS+1), valid_accuracy, 'b' )

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.show()

confuse_mat = plt.matshow(valid_cm.cpu())

plt.show()

print(f1_score(valid_cm))

print(valid_cm.cpu())
import pandas as pd

def predict(it):

    ret = pd.DataFrame(columns=['id', 'target'])

    for epoch, batch in enumerate(it):

        with torch.no_grad():

            output = model(batch.text)

            ret = ret.append(pd.DataFrame({'id':batch.id.tolist(),'target':output.argmax(1).tolist()}))

    return ret



TEST_PATH = "/kaggle/input/nlp-getting-started/test.csv"

SAMPLE_SUBMISSION_PATH = "/kaggle/working/sample_submission.csv"

test_fields = [('id', label_field), ('keyword', None), ('location', None), ('text', text_field)]

test_dataset = data.TabularDataset(TEST_PATH, 'csv', test_fields, skip_header= True)

test_iter = data.Iterator(test_dataset, 16, device=device, train=False, sort=False)



submission = predict(test_iter)

submission.to_csv(SAMPLE_SUBMISSION_PATH, index=False)
import torch as tt

import torch.nn as nn

from torch.autograd import Variable

import torch.nn.functional as F

import torch.optim as optim

from torch.distributions.distribution import Distribution

from torchtext.data import Field, LabelField, BucketIterator, ReversibleField, TabularDataset, BPTTIterator



import numpy as np

import pandas as pd



import os

import string

import time

import random

from sklearn.utils import shuffle



from tqdm import tqdm_notebook as tqdm





import warnings

warnings.simplefilter('ignore')



SEED = 42

np.random.seed(SEED)
with open('../input/bands.csv', encoding='ascii', errors='ignore') as infile:

    df = pd.read_csv(infile)

df.head()
df.shape
df = df.drop_duplicates(subset=['name'])



df.head()

df.shape
df = df.drop(['id', 'country', 'status', 'formed_in', 'genre', 'theme', 'active'], axis = 1)

df.shape
df.head()
df = shuffle(df)

df.head()
band_names = df['name'].tolist()

band_names = '\n'.join(str(x) for x in band_names)

print(len(band_names))

train_set = band_names[:(int(len(band_names)*0.9))]

valid_set = band_names[(int(len(band_names)*0.9)):]

print(len(train_set))

print(len(valid_set))
all_characters = string.printable

def char_tensor(string):

    tensor = tt.zeros(len(string)).long()

    for i in range(len(string)):

        tensor[i] = all_characters.index(string[i])    

    return tensor



def training_sample(chunk_len, batch_size, file):

    input_text = tt.LongTensor(batch_size, chunk_len)

    target = tt.LongTensor(batch_size, chunk_len)

    for i in range(batch_size):

        start_index = random.randint(0, (len(file) - chunk_len))

        finish_index = start_index + chunk_len + 1

        chunk = file[start_index:finish_index]

        input_text[i] = char_tensor(chunk[0:])

        target[i] = char_tensor(chunk[1:])

    return input_text, target



def perplexity(x):

    return 2**x
def train(input_data, target, decoder, optimizer, criterion, curr_epoch):

    decoder.train()

    hidden = decoder.init_hidden(batch_size)

    decoder.zero_grad()

    current_loss, perplexities = 0, list()

    for i in range(chunk_len):

        optimizer.zero_grad()

        output, hidden = decoder(input_data[:,i], hidden)

        loss = criterion(output.view(batch_size, -1), target[:,i])

        perplexities.append(perplexity(loss.item()))

        curr_loss = loss.data.cpu().detach().item()

        current_loss = (i / (i + 1)) * current_loss + (1 - (i / (i + 1))) * curr_loss

    loss.backward(), optimizer.step()

    final_perplexity = np.mean(perplexities)

    return current_loss, final_perplexity





def test(input_data, target, decoder, criterion):

    decoder.eval()

    loss, epoch_loss, perplexities = 0, 0, list()

    hidden = decoder.init_hidden(batch_size)

    with tt.no_grad():

        for i in range(chunk_len):

            output, hidden = decoder(input_data[:,i], hidden)

            loss = criterion(output.view(batch_size, -1), target[:,i])

            perplexities.append(perplexity(loss.item()))

            epoch_loss += loss.data.item()

    final_perplexity = np.mean(perplexities)

    return epoch_loss / chunk_len, final_perplexity

class My_RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):

        super(My_RNN, self).__init__()

        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(input_size, hidden_size)

        self.rnn = nn.GRU(hidden_size, hidden_size, 1)

        self.decoder = nn.Linear(hidden_size, output_size)





    def forward(self, input, hidden):

        batch_size = input.size(0)

        encoded = self.encoder(input)

        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)

        output = self.decoder(output.view(batch_size, -1))

        return output, hidden



    def init_hidden(self, batch_size):

        return Variable(tt.zeros(1, batch_size, self.hidden_size))

def train_nn(decoder, criterion, optimizer, n_epochs=200, early_stopping=0, scheduler=None):

    train_losses, valid_losses = list(), list()

    for epoch in tqdm(range(n_epochs)):

        try:

            train_loss, train_per = train(*training_sample(len(train_set), batch_size, train_set), decoder, optimizer, criterion, epoch)

            valid_loss, valid_per = test(*training_sample(len(valid_set), batch_size, valid_set), decoder, criterion)

            train_losses.append(train_loss)

            valid_losses.append(valid_loss)

            if epoch % 100 == 0 or epoch == n_epochs-1:

                print('Epoch %s | Valid loss %.5f | Train loss %.5f | Valid perplexity %.5f | Train perplexity %.5f' % 

                      (epoch+1, valid_loss, train_loss, train_per, valid_per))

        except:

            continue
batch_size = 64

hidden_size = 128

chunk_len = 512

all_letters = string.printable

num_letters = len(all_letters)



decoder = My_RNN(input_size=len(string.printable),

                  hidden_size=hidden_size, 

                  output_size=len(string.printable))

criterion = nn.CrossEntropyLoss()

optimizer = tt.optim.Adam(decoder.parameters(), lr = 0.01)

train_nn(decoder, criterion, optimizer, n_epochs=1000)
def generate(decoder, prime_str='\n', predict_len=20, temperature=0.8):

    hidden = decoder.init_hidden(1)

    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    predicted = ''

    for p in range(len(prime_str) - 1):

        _, hidden = decoder(prime_input[:,p], hidden)

    input_data = prime_input[:,-1]

    for p in range(predict_len):

        output, hidden = decoder(input_data, hidden)

        output_dist = output.data.view(-1).div(temperature).exp()

        top_i = tt.multinomial(output_dist, 1)[0]

        predicted_char = string.printable[top_i]

        if predicted and predicted_char == '\n':

            break

        else:

            predicted += predicted_char

            input_data = Variable(char_tensor(predicted_char).unsqueeze(0))

    return predicted
for x in range(15):

    print(generate(decoder))

    

tt.save(train_nn, 'death_metal_bands.pt')
import torch

import random

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import numpy as np

import pandas as pd
# Hyperparameters

NUM_EPOCHS = 5000

HIDDEN_SIZE = 100

BATCH_SIZE = 6

LEARNING_RATE = 0.01
# Open dataset

data = open('/kaggle/input/dinosaur-island/dinos.txt', 'r').read()

data = data.lower()

examples = data.split('\n')

chars = sorted(set(' '.join(examples)))

data_size, vocab_size, max_len = len(examples), len(chars), len(max(examples, key=len))

ch_to_idx = {c:i for i, c in enumerate(chars)}

idx_to_ch = {i:c for i, c in enumerate(chars)}

print('Dataset Size: {} Vocabulary Size: {} | Max Length: {}'.format(len(examples), vocab_size, max_len))

print(ch_to_idx)

print(idx_to_ch)
# Construct training set. Y is simply X moved by one place

X = torch.zeros(size=(data_size, max_len, vocab_size))

Y = torch.zeros(size=(data_size, max_len), dtype=torch.long)

print_once = False



# Padding

for i in range(len(examples)):

    while len(examples[i]) < max_len:

        examples[i] += ' '



for example_idx, example in enumerate(examples):

    # Remove last character for input sequence

    text_x = example[:-1]

    for i, char in enumerate(text_x):

        X[example_idx][i][ch_to_idx[char]] = 1



    # Remove first character for target sequence

    text_y = example[1:]

    for i, char in enumerate(text_y):

        Y[example_idx][i] = ch_to_idx[char]

    

    if print_once:

        print(f'Example: {example} | {text_x} {text_y} | X: {X[example_idx]} | Y: {Y[example_idx]}')

        print_once = False



print('X: {} | Y: {}'.format(X.size(), Y.size()))
class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        # RNN Layer

        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=HIDDEN_SIZE, batch_first=True)

        # Fully connected layer

        self.fc = nn.Linear(HIDDEN_SIZE, vocab_size)

    

    def forward(self, x):

        # Initializing hidden state for first input using method defined below

        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)



        # Passing in the input and hidden state into the model and obtaining outputs

        out, hidden = self.rnn(x, hidden)

        

        # Reshaping the outputs such that it can be fit into the fully connected layer

        out = out.contiguous().view(-1, HIDDEN_SIZE)

        out = self.fc(out)

        

        return out, hidden

    

    def init_hidden(self, batch_size):

        # This method generates the first hidden state of zeros which we'll use in the forward pass

        # We'll send the tensor holding the hidden state to the device we specified earlier as well

        hidden = torch.zeros(1, batch_size, HIDDEN_SIZE)

        return hidden



model = Model()

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
for epoch_idx in range(NUM_EPOCHS):

    model.train()

    optimizer.zero_grad()

    output, hidden = model(X)

    loss = loss_fn(output, Y.view(-1).long())

    loss.backward()

    optimizer.step()

    

    if (epoch_idx + 1) % 100 == 0:

        print('Epoch: {} | Loss: {}'.format(epoch_idx + 1, loss))

import string

with torch.no_grad():

    for word in range(10):

        model.eval()

        word = random.choice(string.ascii_lowercase)

        

        for letter_idx in range(max_len):

            start = torch.zeros(size=(1, len(word), vocab_size))

            for w_idx, w in enumerate(word):

                start[0][w_idx][ch_to_idx[w]] = 1

                    

            out, hid = model(start)

            prob = nn.functional.softmax(out[-1], dim=0).data

#             high_prob, char_idx = torch.max(prob, dim=0)

            prob = prob.data.numpy()

            probs = np.array(prob)

            probs /= probs.sum()

            char_idx = np.random.choice(range(0, 27), p=probs)

            letter = idx_to_ch[char_idx.item()]

            if letter == ' ':

                break

            else:            

                word += letter



        print(word)
def predict(model, characters):

    # One-hot encoding our input to fit into the model

    start = torch.zeros(size=(1, len(characters), vocab_size))

    for idx, ch in enumerate(characters):

        start[0][idx][ch_to_idx[ch]] = 1

    out, hidden = model(start)



    prob = nn.functional.softmax(out[-1], dim=0).data

    # Taking the class with the highest probability score from the output

    char_ind = torch.max(prob, dim=0)[1].item()



    return idx_to_ch[char_ind], hidden



def sample(model, start='h'):

    model.eval() # eval mode

    start = start.lower()

    # First off, run through the starting characters

    chars = [ch for ch in start]

    size = 20 - len(chars)

    # Now pass in the previous characters and get a new one

    for ii in range(size):

        char, h = predict(model, chars)

        chars.append(char)



    return ''.join(chars)



sample(model, 'a')
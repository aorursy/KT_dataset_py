from keras.callbacks import LambdaCallback

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM, Dropout

from keras.optimizers import RMSprop

from keras.utils.data_utils import get_file

import numpy as np

import random

import sys

import io

import jieba

import re

text=open('../input/miyuezhuan.txt','r').read(100*1024)

# r=re.compile(r'[：“”，,’‘。！&#@$¥%【】{}？、；<>_…\s（）()\da-zA-Z\-\.]')

# text=r.sub('',text)

lines_of_text=text.split('\n')

lines_of_text=[line.strip() for line in lines_of_text if len(line)>0]

pattern = re.compile(r'\[.*\]')

lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]

pattern = re.compile(r'<.*>')

lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]

pattern = re.compile(r'\.+')

lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]

pattern = re.compile(r' +')

lines_of_text = [pattern.sub("，", lines) for lines in lines_of_text]

pattern = re.compile(r'\\r')

lines_of_text = [pattern.sub("，", lines) for lines in lines_of_text]

text=''.join(lines_of_text)

chars = sorted(list(set(text)))

char_indices = dict((c, i) for i, c in enumerate(chars))

indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40

step = 3

sentences = []

next_chars = []

for i in range(0, len(text) - maxlen, step):

  sentences.append(text[i: i + maxlen])

  next_chars.append(text[i + maxlen])

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)

y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):

  for t, char in enumerate(sentence):

    x[i, t, char_indices[char]] = 1

  y[i, char_indices[next_chars[i]]] = 1

model = Sequential()

model.add(LSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(256))

model.add(Dropout(0.2))

model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

def sample(preds, temperature=1.0):

  # helper function to sample an index from a probability array

  preds = np.asarray(preds).astype('float64')

  preds = np.log(preds) / temperature

  exp_preds = np.exp(preds)

  preds = exp_preds / np.sum(exp_preds)

  probas = np.random.multinomial(1, preds, 1)

  return np.argmax(probas)

def on_epoch_end(epoch, _):

  # Function invoked at end of each epoch. Prints generated text.

  print()

  print('----- Generating text after Epoch: %d' % epoch)



  start_index = random.randint(0, len(text) - maxlen - 1)

  for diversity in [0.2, 0.5, 1.0, 1.2]:

    print('----- diversity:', diversity)



    generated = []

    sentence = text[start_index: start_index + maxlen]

    print('----- Generating with seed: "' + sentence + '"')



    for i in range(400):

      x_pred = np.zeros((1, maxlen, len(chars)))

      for t, char in enumerate(sentence):

        x_pred[0, t, char_indices[char]] = 1.



      preds = model.predict(x_pred, verbose=0)[0]

      next_index = sample(preds, diversity)

      next_char = indices_char[next_index]



      sentence = sentence[1:] + next_char

      generated.append(next_char)

    print('generated---' + sentence+''.join(generated))



print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
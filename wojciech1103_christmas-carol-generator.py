import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import random

import sys



from keras.utils import get_file

from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout

from keras.optimizers import RMSprop
df = pd.read_csv('/kaggle/input/polish-christmas-carols/koledy.csv')
text = ''

for i in range(len(df)):

    text = text + "{} ".format(df.iloc[i]['Lyrics'])
print('Corpus length: ', len(text))
# Length of extracted character sequences

maxlen = 60



# We sample a new sequence every `step` characters

step = 3



# This holds our extracted sequences

sentences = []



# This holds the targets (the follow-up characters)

next_chars = []



for i in range(0, len(text) - maxlen, step):

    sentences.append(text[i: i + maxlen])

    next_chars.append(text[i + maxlen])

print('Number of sequences:', len(sentences))



# List of unique characters in the corpus

chars = sorted(list(set(text)))

print('Unique characters:', len(chars))

# Dictionary mapping unique characters to their index in `chars`

char_indices = dict((char, chars.index(char)) for char in chars)



# Next, one-hot encode the characters into binary arrays.

print('Vectorization...')

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)

y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):

    for t, char in enumerate(sentence):

        x[i, t, char_indices[char]] = 1

    y[i, char_indices[next_chars[i]]] = 1

    
model = Sequential()

model.add(LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))

model.add(LSTM(64, return_sequences=True))

model.add(LSTM(32))

model.add(Dropout(0.1))

model.add(Dense(len(chars), activation='softmax'))
optimizer = RMSprop(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.summary()
def sample(preds, temperature=1.0):

    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)
for epoch in range(1, 20):

    print('epoch', epoch)

    # Fit the model for 1 epoch on the available training data

    model.fit(x, y,

              batch_size=128,

              epochs=3)



    # Select a text seed at random

    start_index = random.randint(0, len(text) - maxlen - 1)

    generated_text = text[start_index: start_index + maxlen]

    

    

    print('--- Generating with seed: "' + generated_text + '"')



    for temperature in [0.2, 0.5, 1.0, 1.2]:

        

        print('------ temperature:', temperature)

        sys.stdout.write(generated_text)

#         text_to_save = text_to_save + '------ temperature: {}\n\n'.format(temperature)

        file_name = 'output_epoch_{}_temp_{}.txt'.format(epoch, temperature)

        text_to_save = ''

        # We generate 400 characters

        for i in range(400):

            sampled = np.zeros((1, maxlen, len(chars)))

            for t, char in enumerate(generated_text):

                sampled[0, t, char_indices[char]] = 1.



            preds = model.predict(sampled, verbose=0)[0]

            next_index = sample(preds, temperature)

            next_char = chars[next_index]



            generated_text += next_char            

            generated_text = generated_text[1:]



            text_to_save = text_to_save + generated_text

            

            sys.stdout.write(next_char)

            sys.stdout.flush()

        print()

        text_file = open(file_name, "w")

        text_file.write("{}".format(text_to_save))

        text_file.close()
import pandas as pd

import numpy as np



from keras.models import Sequential

from keras.layers import Embedding, Dense, Activation, Dropout

from keras.layers.normalization import BatchNormalization

from keras.layers import GRU

from keras.optimizers import RMSprop



import random
num_questions = 500
questions = pd.read_csv("../input/Questions.csv", encoding='latin1')

text = list(questions["Body"])[:num_questions]
chars = set()

for i in range(len(text)):

    chars = chars.union(set(text[i]))

chars = sorted(chars)

char_index = dict((c, i) for i, c in enumerate(chars))

index_char = dict((i, c) for i, c in enumerate(chars))
sequence_len = 30

step = 1



num_sequences = 0



for quest in text:

    for i in range(0, len(quest) - sequence_len, step):

        num_sequences += 1

        #sequences.append(quest[i: i + sequence_len])

        #next_chars.append(quest[i + sequence_len])

        

print("# sequences:", num_sequences)





X = np.zeros((num_sequences, sequence_len, len(chars)), dtype = np.bool)

y = np.zeros((num_sequences, len(chars)), dtype = np.bool)



for quest in text:

    for i in range(0, len(quest) - sequence_len, step):

        seq = quest[i: i + sequence_len]

        next_char = quest[i + sequence_len]

        for j, ch in enumerate(seq):

            X[i, j, char_index[ch]] = 1

        y[i, char_index[next_char]] = 1
model = Sequential()



model.add(GRU(512, return_sequences = True, input_shape = (sequence_len, len(chars))))

#model.add(BatchNormalization())

model.add(Dropout(.2))



model.add(GRU(256))

#model.add(BatchNormalization())

model.add(Dropout(.2))



model.add(Dense(len(chars)))

model.add(Activation('softmax'))



optimizer = RMSprop(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)
def sample(preds, temperature=1.0):

    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)





def data_generator(batch_size):

    while 1:

        index = np.random.randint(0, num_sequences, batch_size)

        yield X[index, :, :], y[index, :]
n_iter = 5

batch_size = 64

max_gen_len = 300

n_samples = 10





for iteration in range(1, n_iter):

    print("Iteration:", iteration)

    

    model.fit_generator(data_generator(batch_size), 

                        samples_per_epoch = batch_size * n_samples, 

                        nb_epoch = 1, verbose=1)

    

    seq_index = random.randint(0, len(text))

    start_index = random.randint(0, len(text[seq_index]) - sequence_len - 1)

    print("Sequence seed:\n\t", text[seq_index][start_index : start_index + sequence_len])

    

    for temper in [.2, .6, 1.3]:

        print("-" * 30)

        generated = ''

        sequence = text[seq_index][start_index : start_index + sequence_len]

        generated += sequence

        

        for i in range(max_gen_len):

            x = np.zeros((1, sequence_len, len(chars)))

            for i, ch in enumerate(sequence):

                    x[0, i, char_index[ch]] = 1.

                    

            pred_ch = model.predict(x, verbose = 0)[0]

            next_index = sample(pred_ch, temper)

            next_char = index_char[next_index]



            generated += next_char

            sequence = sequence[1:] + next_char

        print(generated)

    print("-" * 30)
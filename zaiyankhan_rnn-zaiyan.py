from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Embedding

from keras.models import Sequential

from numpy import array

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences
def generate(model, tokenizer, max_length, seed_text, n_words):

    in_text = seed_text

    for _ in range(n_words):

        encoded = tokenizer.texts_to_sequences([in_text])[0]

        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')

        yhat = model.predict_classes(encoded, verbose=0)

        out_word = ''

        for word, index in tokenizer.word_index.items():

            if index == yhat:

                out_word = word

                break

        in_text += ' ' + out_word

    return in_text

 

data = """ All the world's a stage, And all the men and women merely players; They have their exits and their entrances, And one man in his time plays many parts, His acts being seven ages. At first, the infant, Mewling and puking in the nurse's arms. Then the whining schoolboy, with his satchel And shining morning face, creeping like snail Unwillingly to school. And then the lover, Sighing like furnace, with a woeful ballad Made to his mistress' eyebrow. Then a soldier, Full of strange oaths and bearded like the pard, Jealous in honor, sudden and quick in quarrel, Seeking the bubble reputation Even in the cannon's mouth. And then the justice, In fair round belly with good capon lined, With eyes severe and beard of formal cut, Full of wise saws and modern instances; And so he plays his part. The sixth age shifts Into the lean and slippered pantaloon, With spectacles on nose and pouch on side; His youthful hose, well saved, a world too wide For his shrunk shank, and his big manly voice, Turning again toward childish treble, pipes And whistles in his sound. Last scene of all, That ends this strange eventful history, Is second childishness and mere oblivion, Sans teeth, sans eyes, sans taste, sans everything. """



tokenizer = Tokenizer()

tokenizer.fit_on_texts([data])

encoded = tokenizer.texts_to_sequences([data])[0]

vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)

sequences = list()

for i in range(2, len(encoded)):

	sequence = encoded[i-2:i+1]

	sequences.append(sequence)

print('Total Sequences: %d' % len(sequences))

max_length = max([len(seq) for seq in sequences])

sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

print('Max Sequence Length: %d' % max_length)

sequences = array(sequences)

X, y = sequences[:,:-1],sequences[:,-1]

y = to_categorical(y, num_classes=vocab_size)



#LSTM Model

model = Sequential()

model.add(Embedding(vocab_size, 10, input_length=max_length-1))

model.add(LSTM(50))

model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=170, verbose=2)


# Evaluation

pword = generate(model, tokenizer, max_length-1, 'men and women', 1)

print("\nPredicting a word based on 3 consecutive words")

print("Input (3 words) : men and women")

print("Output : " + pword)



pgword = generate(model, tokenizer, max_length-1, '', 100)

print("\n Generating text based on trained RNN :")

print(pgword)
import keras
from keras import layers
import sys
import numpy as np
# seed or initial text
text = 'this text file can be any text, as long as it contains text longer than maxlen defined below'
chars = ['\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
         '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a',
         'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
         'x', 'y', 'z', '{', '|', '}', '~']
print('initial text length: ', len(text))
print('vocabulary length: ', len(chars))
# Dictionary mapping unique characters to their index in chars
char_indices = dict((char, chars.index(char)) for char in chars)
pre_trained_model = keras.utils.get_file(
    'pre-trained.hdf5',
    origin='https://github.com/Tony607/Yelp_review_generation/releases/download/V0.1/pre-trained.hdf5')
maxlen = 60

model = keras.models.Sequential()
model.add(layers.LSTM(1024, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(layers.LSTM(1024, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
model.load_weights(pre_trained_model)

optimizer = keras.optimizers.Adam(lr=0.0002)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
# temperature: from 0 to 1
def sample(preds, temperature):
    preds = np.asarray(preds).astype('float64')
    vectfunc = np.vectorize(lambda p: p + (1e-10) if p == 0 else p)
    preds = vectfunc(preds)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
def random_reviews(num_chars, temperature):
    # sample a start index
    start_index = np.random.randint(0, len(text) - maxlen - 1)
    # the initial sampled text with maxlen long
    generated_text = text[start_index: start_index + maxlen]
    print('Coming up with several reviews for you...')

#     sys.stdout.write(generated_text)

    for i in range(num_chars):
        sampled = np.zeros((1, maxlen, len(chars)))
        # one-hot encoding: Turn each char to char index.
        for t, char in enumerate(generated_text):
            sampled[0, t, char_indices[char]] = 1.
        # Predict next char probabilities
        preds = model.predict(sampled, verbose=0)[0]
        # Add some randomness by sampling given probabilities.
        next_index = sample(preds, temperature)
        # Turn char index to char.
        next_char = chars[next_index]
        # Append char to generated text string
        generated_text += next_char
        # Pop the first char in generated text string.
        generated_text = generated_text[1:]
        # Print the new generated char.
        sys.stdout.write(next_char)
        sys.stdout.flush()
random_reviews(500, 0.9)
random_reviews(500, 0.5)
random_reviews(500, 0.1)

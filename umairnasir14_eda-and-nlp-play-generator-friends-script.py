# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from keras.preprocessing import sequence
import keras
import tensorflow as tf
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from PIL import Image
from wordcloud import WordCloud
import nltk
import spacy
from nltk import word_tokenize
from nltk.util import ngrams
import re
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

text = ""
folder_name = "../input/friends-tv-series-screenplay-script/"
for f in glob.glob(folder_name + '/*.txt'):
    temp = open(f,'r')    
    text += temp.read()
    temp.close()
print ('Length of text: {} characters'.format(len(text)))
print(text[:500])
# adding screenplay notes to stopwords
nlp = spacy.load("en")
nlp.Defaults.stop_words |= {"d","ll","m","re","s","ve", "t", "oh", "uh", "na", "okay",
                           "didn","don","gon","j","hm","um","dr","room","int", "ext", 
                           "cut", "day", "night", "theme", "tune","music", "ends","view","opening credits scene", 
                            "commercial break scene", "hey hey hey", "hey", "closing credits scene","scene",
                            "closeup", 'freshly', 'squeezed', 'fade'}
stopwords = nlp.Defaults.stop_words
all_words = nltk.tokenize.word_tokenize(text.lower())
all_words_no_stop = nltk.FreqDist(w.lower() for w in all_words if w not in stopwords)
def color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(1, 90%, 47%)"

wc = WordCloud(background_color="black", max_words=1000,
               stopwords=stopwords, contour_width=4, contour_color='steelblue')

wc.generate(" ".join(all_words_no_stop.keys()))

plt.figure(figsize=(18, 10))
plt.imshow(wc.recolor(color_func=color_func, random_state=3),interpolation="bilinear")
plt.axis("off");
print("Chandler has been called {} times".format(all_words_no_stop['chandler']))
print("Joey has been called {} times".format(all_words_no_stop['joey']))
print("Monica has been called exactly {} times".format(all_words_no_stop['monica']))
print("Ross has been called exactly {} times".format(all_words_no_stop['ross']))
print("Rachel has been called exactly {} times".format(all_words_no_stop['rachel']))
print("Phoebe has been called exactly {} times".format(all_words_no_stop['phoebe']))
print("{} times Joey used how you doin?".format(all_words_no_stop['doin']))
print("Joey asked for sandwiches {} times".format(all_words_no_stop['sandwich']))
vocab = sorted(set(text))
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
def text_to_int(text):
  return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)
#extra step, better to do it
def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

print(int_to_text(text_as_int[:13]))

print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))
seq_length = 1000  # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):  # for the example: hello
    input_text = chunk[:-1]  # hell
    target_text = chunk[1:]  # ello
    return input_text, target_text  # hell, ello

dataset = sequences.map(split_input_target)  # we use map to apply the above function to every entry
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()
for input_example_batch, target_example_batch in data.take(1):
  example_batch_predictions = model(input_example_batch)  
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  # print out the output shape

print(len(example_batch_predictions))
print(example_batch_predictions)

pred = example_batch_predictions[0]
print(len(pred))
print(pred)
# notice this is a 2d array of length 1000, where each interior array is the prediction for the next character at each time step

time_pred = pred[0]
print(len(time_pred))
print(time_pred)
# If we want to determine the predicted character we need to sample the output distribution (pick a value based on probabillity)
sampled_indices = tf.random.categorical(pred, num_samples=1)

# now we can reshape that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
predicted_chars = int_to_text(sampled_indices)

predicted_chars  # and this is what the model predicted for training sequence 1
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
history = model.fit(data, epochs=80, callbacks=[checkpoint_callback])
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
#checkpoint_num = 10
#model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
#model.build(tf.TensorShape([1, None]))
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1500

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
    
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))
inp = input("Type a starting string: ")
print(generate_text(model, inp))
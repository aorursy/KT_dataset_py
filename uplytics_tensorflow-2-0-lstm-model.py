!pip install tensorflow_datasets



import tensorflow as tf

import tensorflow_datasets as tfds

import os
mbti_dataset_line = tf.data.TextLineDataset("../input/mbti-clean-with-categories/mbti_clean.csv")
for ex in mbti_dataset_line.take(5):

  print(ex)
def label(line):

  label =  tf.strings.substr([line],[-10],[1])

  if label[0]==',':

    label = tf.strings.substr([line],[-9],[1])

  else:

    label = tf.strings.substr([line],[-10],[2])

  labelnum=tf.strings.to_number(label,tf.int64)

  line= tf.strings.substr([line],[6],(tf.strings.length([line])-17))

  return line[0], labelnum[0]



mbti_dataset_line = mbti_dataset_line.skip(1).map(lambda line: label(line))

for ex in mbti_dataset_line.take(5):

  print(ex)
BUFFER_SIZE = 100000

BATCH_SIZE = 64

TAKE_SIZE = 5000
mbti_dataset_line = mbti_dataset_line.shuffle(

    BUFFER_SIZE, reshuffle_each_iteration=False)
tokenizer = tfds.features.text.Tokenizer()



vocabulary_set = set()

for text_tensor, _ in mbti_dataset_line:

  some_tokens = tokenizer.tokenize(text_tensor.numpy())

  vocabulary_set.update(some_tokens)



vocab_size = len(vocabulary_set)

vocab_size
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
example_text = next(iter(mbti_dataset_line))[0].numpy()

print(example_text)
encoded_example = encoder.encode(example_text)

print(encoded_example)
def encode(text_tensor, label):

  encoded_text = encoder.encode(text_tensor.numpy())

  return encoded_text, label



def encode_map_fn(text, label):

  return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))



all_encoded_data = mbti_dataset_line.map(encode_map_fn)
train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)

train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))



test_data = all_encoded_data.take(TAKE_SIZE)

test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))
sample_text, sample_labels = next(iter(test_data))



sample_text[0], sample_labels[0]
vocab_size += 1
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size+1, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
# One or more dense layers.

# Edit the list in the `for` line to experiment with layer sizes.

for units in [64, 64]:

  model.add(tf.keras.layers.Dense(units, activation='relu'))



# Output layer. The first argument is the number of labels.

model.add(tf.keras.layers.Dense(16, activation='softmax'))
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_data, epochs=1, validation_data=test_data)
eval_loss, eval_acc = model.evaluate(test_data)



print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
exit()
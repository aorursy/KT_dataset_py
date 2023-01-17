# Selecting Tensorflow version v2 (the command is relevant for Colab only).

# %tensorflow_version 2.x
import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import platform

import random

import math

import datetime



print('Python version:', platform.python_version())

print('Tensorflow version:', tf.__version__)

print('Keras version:', tf.keras.__version__)
# Load the TensorBoard notebook extension.

# %reload_ext tensorboard

%load_ext tensorboard
dataset_size = 5000

sequence_length = 2

max_num = 100
# Generates summation sequences and summation results in form of vector if numbers.

def generate_sums(dataset_size, sequence_length, max_num):

    # Initial dataset states.

    x, y = [], []

    

    # Generating sums.

    for i in range(dataset_size):

        sequence = [random.randint(1, max_num) for _ in range(sequence_length)]

        x.append(sequence)

        y.append(sum(sequence))

    

    return x, y
x_train, y_train = generate_sums(

    dataset_size=dataset_size,

    sequence_length=sequence_length,

    max_num=max_num

)



print('x_train:\n', x_train[:3])

print()

print('y_train:\n', y_train[:3])
# Convert array of numbers for x and y into strings.

# Also it adds a space (" ") padding to strings to make them of the same length. 

def dataset_to_strings(x, y, max_num):

    # Initial dataset states.

    x_str, y_str = [], []

    

    sequnce_length = len(x[0])

    

    # Calculate the maximum length of equation (x) string (i.e. of "11+99")

    num_of_pluses = sequnce_length - 1

    num_of_chars_per_digit = math.ceil(math.log10(max_num + 1))

    max_x_length = sequnce_length * num_of_chars_per_digit + num_of_pluses

    

    # Calculate the maximum length of label (y) string (i.e. of "167")

    max_y_length = math.ceil(math.log10(sequnce_length * (max_num + 1)))

    

    # Add a space " " padding to equation strings to make them of the same length.

    for example in x:

        str_example = '+'.join([str(digit) for digit in example])

        str_example += ''.join([' ' for padding in range(max_x_length - len(str_example))])

        x_str.append(str_example)

    

    # Add a space " " padding to labels strings to make them of the same length.

    for label in y:

        str_example = str(label)

        str_example += ''.join([' ' for padding in range(max_y_length - len(str_example))])

        y_str.append(str_example)

    

    return x_str, y_str
x_train_str, y_train_str = dataset_to_strings(x_train, y_train, max_num)



print('x_train_str:\n', np.array(x_train_str[:3]))

print()

print('y_train_str:\n', np.array(y_train_str[:3]))
# Since we allow only numbers, plus sign and spaces the vocabulary looks pretty simple.

vocabulary = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
# Python dictionary that will convert a character to its index in the vocabulary.

char_to_index = {char: index for index, char in enumerate(vocabulary)}



print(char_to_index)
# Converts x and y arrays of strings into array of char indices.

def dataset_to_indices(x, y, vocabulary):

    x_encoded, y_encoded = [], []

    

    char_to_index = {char: index for index, char in enumerate(vocabulary)}

    

    for example in x:

        example_encoded = [char_to_index[char] for char in example]

        x_encoded.append(example_encoded)

        

    for label in y:

        label_encoded = [char_to_index[char] for char in label]

        y_encoded.append(label_encoded)

        

    return x_encoded, y_encoded
x_train_encoded, y_train_encoded = dataset_to_indices(

    x_train_str,

    y_train_str,

    vocabulary

)



print('x_train_encoded:\n', np.array(x_train_encoded[:3]))

print()

print('y_train_encoded:\n', np.array(y_train_encoded[:3]))
# Convert x and y sets of numbers into one-hot vectors.

def dataset_to_one_hot(x, y, vocabulary):

    x_encoded, y_encoded = [], []

    

    for example in x:

        pattern = []

        for index in example:

            vector = [0 for _ in range(len(vocabulary))]

            vector[index] = 1

            pattern.append(vector)

        x_encoded.append(pattern)

            

    for label in y:

        pattern = []

        for index in label:

            vector = [0 for _ in range(len(vocabulary))]

            vector[index] = 1

            pattern.append(vector)

        y_encoded.append(pattern)

        

    return x_encoded, y_encoded
x_train_one_hot, y_train_one_hot = dataset_to_one_hot(

    x_train_encoded,

    y_train_encoded,

    vocabulary

)



print('x_train_one_hot:\n', np.array(x_train_one_hot[:1]))

print()

print('y_train_one_hot:\n', np.array(y_train_one_hot[:1]))
# Generates a dataset.

def generate_dataset(dataset_size, sequence_length, max_num, vocabulary):

    # Generate integet sum sequences.

    x, y = generate_sums(dataset_size, sequence_length, max_num)

    # Convert integer sum sequences into strings.

    x, y = dataset_to_strings(x, y, max_num)

    # Encode each character to a char indices.

    x, y = dataset_to_indices(x, y, vocabulary)

    # Encode each index into one-hot vector.

    x, y = dataset_to_one_hot(x, y, vocabulary)

    # Return the data.

    return np.array(x), np.array(y)
x, y = generate_dataset(

    dataset_size,

    sequence_length,

    max_num,

    vocabulary

)



print('x:\n', x[:1])

print()

print('y:\n', y[:1])
print('x.shape: ', x.shape) # (input_sequences_num, input_sequence_length, supported_symbols_num)

print('y.shape: ', y.shape) # (output_sequences_num, output_sequence_length, supported_symbols_num)
# How many characters each summation expression has.

input_sequence_length = x.shape[1]



# How many characters the output sequence of the RNN has.

output_sequence_length = y.shape[1]



# The length of one-hot vector for each character in the input (should be the same as vocabulary_size).

supported_symbols_num = x.shape[2]



# The number of different characters our RNN network could work with (i.e. it understands only digits, "+" and " ").

vocabulary_size = len(vocabulary)



print('input_sequence_length: ', input_sequence_length)

print('output_sequence_length: ', output_sequence_length)

print('supported_symbols_num: ', supported_symbols_num)

print('vocabulary_size: ', vocabulary_size)
# Converts a sequence (array) of one-hot encoded vectors back into the string based on the provided vocabulary.

def decode(sequence, vocabulary):

    index_to_char = {index: char for index, char in enumerate(vocabulary)}

    strings = []

    for char_vector in sequence:

        char = index_to_char[np.argmax(char_vector)]

        strings.append(char)

    return ''.join(strings)
decode(y[0], vocabulary)
epochs_num = 200

batch_size = 128
model = tf.keras.models.Sequential()



# Encoder

# -------



model.add(tf.keras.layers.LSTM(

    units=128,

    input_shape=(input_sequence_length, vocabulary_size),

    recurrent_initializer=tf.keras.initializers.GlorotNormal()

))



# Decoder

# -------



# We need this layer to match the encoder output shape with decoder input shape.

# Encoder outputs ONE vector of numbers but for decoder we need have output_sequence_length vectors.

model.add(tf.keras.layers.RepeatVector(

    n=output_sequence_length,

))



model.add(tf.keras.layers.LSTM(

    units=128,

    return_sequences=True,

    recurrent_initializer=tf.keras.initializers.GlorotNormal()

))



model.add(tf.keras.layers.TimeDistributed(

    layer=tf.keras.layers.Dense(

        units=vocabulary_size,

    )

))



model.add(tf.keras.layers.Activation(

    activation='softmax'

))
model.summary()
tf.keras.utils.plot_model(

    model,

    show_shapes=True,

    show_layer_names=True,

)
log_dir=".logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(

    optimizer=adam_optimizer,

    loss=tf.keras.losses.categorical_crossentropy,

    metrics=['accuracy'],

)
history = model.fit(

    x=x,

    y=y,

    epochs=epochs_num,

    batch_size=batch_size,

    validation_split=0.1,

    callbacks=[tensorboard_callback]

)
# Renders the charts for training accuracy and loss.

def render_training_history(training_history):

    loss = training_history.history['loss']

    val_loss = training_history.history['val_loss']



    accuracy = training_history.history['accuracy']

    val_accuracy = training_history.history['val_accuracy']



    plt.figure(figsize=(14, 4))



    plt.subplot(1, 2, 1)

    plt.title('Loss')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.plot(loss, label='Training set')

    plt.plot(val_loss, label='Test set', linestyle='--')

    plt.legend()

    plt.grid(linestyle='--', linewidth=1, alpha=0.5)



    plt.subplot(1, 2, 2)

    plt.title('Accuracy')

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.plot(accuracy, label='Training set')

    plt.plot(val_accuracy, label='Test set', linestyle='--')

    plt.legend()

    plt.grid(linestyle='--', linewidth=1, alpha=0.5)



    plt.show()
render_training_history(history)
x_test, y_test = generate_dataset(dataset_size, sequence_length, max_num, vocabulary)



print('x_test:\n', x_test[:1])

print()

print('y_test:\n', y_test[:1])
predictions = model.predict(x_test)



print('predictions.shape: ', predictions.shape)

print()

print('predictions[0]:\n', predictions[0])

print()

print('predictions[1]:\n', predictions[1])
x_encoded = [decode(example, vocabulary) for example in x_test]

y_expected = [decode(label, vocabulary) for label in y_test]

y_predicted = [decode(prediction, vocabulary) for prediction in predictions]



explore_num = 40

for example, label, prediction in list(zip(x_encoded, y_expected, y_predicted))[:explore_num]:

    checkmark = '✓' if label == prediction else ''

    print('{} = {} [predict: {}] {}'.format(example, label, prediction, checkmark))
%tensorboard --logdir .logs/fit
model_name = 'numbers_summation_rnn.h5'

model.save(model_name, save_format='h5')
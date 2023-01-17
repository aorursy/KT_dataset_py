import numpy as np

import matplotlib.pyplot as plt



from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import ModelCheckpoint

from keras.initializers import he_normal
# Loading the data

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

print(train_data[0])
# For shits and giggles - decoding words from numeric back to english

word_index = imdb.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

decoded_review
# Preparing data

# Encoding the integer sequences into a binary matrix

def vectorize_sequences(sequences, dimension  = 10000):

    # Create an all-zero matrix of shape (len(sequences), dimension)

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        # Sets specific indicies of results[i] to 1s

        results[i, sequence] = 1.

    return results



# Vectorize data

x_train = vectorize_sequences(train_data)

x_test = vectorize_sequences(test_data)



# Vectorize targets

y_train = np.asarray(train_labels).astype('float32')

y_test = np.asarray(test_labels).astype('float32')
x_train[0]
y_train
# Function to plot learning curves

def lc():

    H = model.history.history

    fig = plt.figure(figsize = (19, 6))

    plt.subplot(121)

    plt.plot(H['accuracy'], label = 'acc')

    

    try:

        plt.plot(H['val_accuracy'], label = 'val_acc')

    except:

        pass

    

    plt.grid(); plt.legend()

    

    plt.subplot(122)

    plt.plot(H['loss'], label = 'loss')

    

    try:

        plt.plot(H['val_loss'], label = 'val_loss')

    except:

        pass

    

    plt.grid(); plt.legend()

    plt.show()
# Train model on 15000 samples and validate on 10000

model = Sequential()

model.add(Dense(16, activation = 'relu'))

model.add(Dense(16, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])



model.fit(x_train, y_train, epochs = 20, batch_size = 512, validation_split = 0.4, verbose = 0)



# Learning curves

lc()



# Overfitting
# Train model on 4 epochs and full train data and evaluate on test data

model = Sequential()

model.add(Dense(16, activation = 'relu'))

model.add(Dense(16, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])



model.fit(x_train, y_train, epochs = 4, batch_size = 512, verbose = 0)



results = model.evaluate(x_test, y_test)

print(f'Test data loss: {results[0]}\nTest data accuracy: {results[1]}')



# Learning curves

lc()
def create_model(layers, act = 'relu', output_act = 'sigmoid', loss = 'binary_crossentropy'):

    init = he_normal(seed = 666)

    model = Sequential()

    for layer in layers:

        model.add(Dense(layer, activation = act, kernel_initializer = init))

    model.add(Dense(1, activation = output_act, kernel_initializer = init))

    

    model.compile(optimizer = 'rmsprop', loss = loss, metrics = ['accuracy'])

    return model



def evaluate():

    results = model.evaluate(x_test, y_test)

    print(f'Test data loss: {results[0]}\nTest data accuracy: {results[1]}')

    return results
# Setting up experiments

# Dict key notation: [hidden layers], activation of hidden layers, activation of output layer, loss function

experiments = {'[16], relu, sigm, bc': create_model([16]),

               '[16, 16], relu, sigm, bc': create_model([16, 16]),

               '[16, 16, 16], relu, sigm, bc': create_model([16, 16, 16]),

               '[32], relu, sigm, bc': create_model([32]),

               '[32, 32], relu, sigm, bc': create_model([32, 32]),

               '[32, 32, 32], relu, sigm, bc': create_model([32, 32, 32]),

               '[64], relu, sigm, bc': create_model([64]),

               '[64, 64], relu, sigm, bc': create_model([64, 64]),

               '[64, 64, 64], relu, sigm, bc': create_model([64, 64, 64]),

               '[16, 16], relu, sigm, mse': create_model([16, 16], loss = 'mse'),

               '[16, 16], tanh, sigm, bc': create_model([16, 16], act = 'tanh')}



# Dictionary to store evaluation results

results = dict()



# Loo through all experiments

for experiment in experiments:

    print(experiment)

    

    model = experiments[experiment]

    checkpoint = ModelCheckpoint('best_model.hdf5', save_best_only = True, save_weights_only = True)

    model.fit(x_train, y_train, epochs = 10, batch_size = 512, validation_split = 0.4, verbose = 0, callbacks = [checkpoint])

    

    # Load best weights

    model.load_weights('best_model.hdf5')

    

    results[experiment] = evaluate()

    lc()
fig = plt.figure(figsize = (19, 6))

for r in results:

    plt.bar(r, results[r][1], label = r)

plt.xticks([], [])

plt.legend(loc = 4); plt.grid()

plt.show()
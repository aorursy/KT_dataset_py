import numpy as np # Linear algebra

import matplotlib.pyplot as plt # Graphs

%matplotlib inline
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam
data = np.loadtxt('../input/mushrooms.csv', dtype=bytes, delimiter=',', skiprows=1).astype("str")

nb_mushrooms = data.shape[0]

print(data.shape)

print(data[:5, :10])
def get_code_book(data):

    code_book = []

    for j in range(data.shape[1]):

        code = {}

        num = 0

        for i in range(data.shape[0]):

            if data[i, j] not in code:

                code[data[i, j]] = num

                num += 1

        code_book.append(code)

    return code_book

CODE_BOOK = get_code_book(data)
def translate(data, code_book):

    result = np.empty(data.shape, dtype = 'int')

    for i in range(data.shape[0]):

        for j in range(data.shape[1]):

            result[i,j] = CODE_BOOK[j][data[i, j]]

    return result

num_data = translate(data, CODE_BOOK)
CATEG = [len(code) for code in CODE_BOOK[1:]]

INPUT_SIZE = sum(CATEG)

print(INPUT_SIZE)
all_labels = num_data[:, 0]

all_features = num_data[:, 1:]



idx = np.arange(0, num_data.shape[0])

np.random.shuffle(idx)



n_val = nb_mushrooms // 10

n_train = nb_mushrooms = n_val



val_features = all_features[idx[:n_val], :]

train_features = all_features[idx[n_val:], :]

val_labels = all_labels[idx[:n_val]]

train_labels = all_labels[idx[n_val:]]
def make_one_hot_row(data_row):

    result = np.zeros((INPUT_SIZE, ), dtype = "float32")

    offset = 0

    for j in range(data_row.shape[0]):

        result[offset + data_row[j]] = 1.0

        offset += CATEG[j]

    return result
def make_one_hot(data):

    result = np.zeros((data.shape[0], INPUT_SIZE), dtype="float32")

    for i in range(data.shape[0]):

        result[i,:] = make_one_hot_row(data[i, :])

    return result
model = Sequential()

model.add(Dense(32, input_shape=[INPUT_SIZE]))

model.add(Dense(32))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(make_one_hot(train_features), train_labels,

          validation_data=(make_one_hot(val_features), val_labels),

          epochs=5,

          verbose=2)
y_hat = model.predict(make_one_hot(val_features))

plt.scatter(val_labels, y_hat)
def is_edible(list_of_strings, cutoff=0.5):

    if len(list_of_strings) != len(CATEG): 

        print("Error, we need {} parameter values!".format(len(CATEG)))

        return

    data_row = np.zeros(len(CATEG), dtype='int')

    for j in range(len(list_of_strings)):

        if not list_of_strings[j] in CODE_BOOK[j+1]:

            print("Error, parameter value {} is not recognized!".format(list_of_strings[j]))

            return

        data_row[j] = CODE_BOOK[j+1][list_of_strings[j]]

    features = make_one_hot_row(data_row).reshape(1,INPUT_SIZE)

    e = model.predict(features)

    if e > cutoff:

        return True

    else:

        return False
for i in range(5):

    print(is_edible(data[i, 1:]))
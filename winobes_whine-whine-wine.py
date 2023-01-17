# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split



df = pd.read_csv('../input/winemag-data-130k-v2.csv')



tokenizer = Tokenizer()

tokenizer.fit_on_texts(list(df['description']))

X = tokenizer.texts_to_sequences(list(df['description']))

sequence_len = max(map(len, X))

X = pad_sequences(X)

vocab_size = len(tokenizer.word_index) + 1



label_encoder = LabelEncoder()

y = label_encoder.fit_transform(list(df['taster_name']))

one_hot_encoder = OneHotEncoder(sparse=False)

y = one_hot_encoder.fit_transform(y.reshape(len(y), 1))

num_labels = y.shape[1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=123)
from keras.layers import Dense, Input, Flatten, Dropout

from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D, Activation, Dense

from keras.models import Sequential



model = Sequential()



# simplest nn you can imagine

# model.add(Dense(300, input_shape=(sequence_len,), activation='relu'))

# model.add(Dense(num_labels, activation='softmax'))



# simple cnn

model.add(Embedding(vocab_size, 200, input_length=sequence_len))

model.add(Conv1D(3, 5, activation='relu', input_shape=(sequence_len,)))

model.add(GlobalMaxPooling1D())

model.add(Dense(num_labels, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, validation_split=0.1, epochs=1)
from sklearn.metrics import precision_recall_fscore_support as score



predictions = model.predict(X_test)



predictions = np.argmax(predictions, axis=1)

actual = np.argmax(y_test, axis=1)

res = pd.DataFrame(np.column_stack(score(actual, predictions)), columns=['precision', 'recall', 'f-score', 'support'])
res
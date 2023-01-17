import numpy as np

import pandas as pd

from tensorflow import set_random_seed

from sklearn.model_selection import train_test_split

from keras.layers import Embedding, LSTM, Dropout, Dense

from keras.models import Sequential

from nltk.tokenize import word_tokenize



np.random.seed(0)

set_random_seed(0)
data = pd.read_json('../input/Sarcasm_Headlines_Dataset.json', lines=True)

data.head()
start = '##START##'

end = '##END##'

ukc = '##UKC##'

pad = '##PAD##'

data['headline_modified'] = data['headline'].apply(word_tokenize)

data['headline_modified'] = data['headline_modified'].apply(lambda x:

                                                            list(filter(lambda y: y.isalpha(), x)))

l = max([len(h) for h in data['headline_modified']])

data['headline_modified'] = data['headline_modified'].apply(lambda x: [start] + x + [end] +

                                                            [pad for i in range(l - len(x))])

X_train, X_test, y_train, y_test = train_test_split(data['headline_modified'], data['is_sarcastic'],

                                                    test_size=0.2, random_state=0)
words = {}

words_count = {}

words[ukc] = 0

c = 1

for h in X_train:

    for w in h:

        words_count[w] = words_count.get(w, 0) + 1

        if w not in words and words_count[w] > 1:

            words[w] = c

            c += 1
len(words)
rev_words = {}

for key, value in words.items():

    rev_words[value] = key
for i, h in enumerate(X_train):

    X_train.iloc[i] = np.array([words.get(w, words[ukc]) for w in h])

for i, h in enumerate(X_test):

    X_test.iloc[i] = np.array([words.get(w, words[ukc]) for w in h])
for h in X_train[:10]:

    print(len(h))
model = Sequential()

model.add(Embedding(len(words), 32, input_length=l + 2))

model.add(LSTM(50))

model.add(Dropout(0.2, seed=0))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.summary()
X_train = np.array(X_train.tolist())

X_test = np.array(X_test.tolist())
model.fit(x=X_train, y=y_train, epochs=5)
model.evaluate(x=X_test, y=y_test)
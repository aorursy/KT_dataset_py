import numpy as np

RANDOM_SEED = 4

np.random.seed(RANDOM_SEED) # set random seed for reproducability
import pandas as pd

df_neut = pd.read_csv("../input/wikipedia-promotional-articles/good.csv")

df_prom = pd.read_csv("../input/wikipedia-promotional-articles/promotional.csv")
df_prom = df_prom.drop(df_prom.columns[1:], axis=1)

df_neut = df_neut.drop(df_neut.columns[1:], axis=1)
df_neut.head()
df_prom.head()
df_neut.insert(1, 'label', 0) # neutral labels

df_prom.insert(1, 'label', 1) # promotional labels
df_prom.head()
df_neut.head()
df = pd.concat((df_neut, df_prom), ignore_index=True, axis=0) # merge dataframes
df.head()
df = df.reindex(np.random.permutation(df.index))

df.head()
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
from keras.preprocessing.text import Tokenizer



# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 50000



text_data = [str(txt) for txt in df_train['text'].values] # convert text data to strings

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True) # create tokenizer object

tokenizer.fit_on_texts(text_data) # make dictionary



x_train = tokenizer.texts_to_sequences(text_data) # vectorize dataset
from keras.preprocessing import sequence



# Max number of words in each sequence

MAX_SEQUENCE_LENGTH = 400



x_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
y_train = df_train['label'].values
from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout

from keras.layers.embeddings import Embedding

from keras.optimizers import Adam



model = Sequential()
EMBEDDING_DIM = 100

model.add(Embedding(MAX_NB_WORDS+1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(80))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
EPOCHS = 2

BATCH_SIZE = 64



history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.15)
x_test = np.array(tokenizer.texts_to_sequences([str(txt) for txt in df_test['text'].values]))

x_test = sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)



y_test = df_test['label'].values
scores = model.evaluate(x_test, y_test, batch_size=128)

print("The model has a test loss of %.2f and a test accuracy of %.1f%%" % (scores[0], scores[1]*100))
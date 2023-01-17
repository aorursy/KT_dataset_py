import numpy as np

import pandas as pd



from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Embedding, SimpleRNN
# Read data from file, parse and split it into raw pandas dataframes

df1 = pd.read_json("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)

df1 = df1.loc[:, ['headline', 'is_sarcastic']]

df1['headline'] = df1['headline'].str.split()



df2 = pd.read_json("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines=True)

df2 = df2.loc[:, ['headline', 'is_sarcastic']]

df2['headline'] = df2['headline'].str.split()



# Join the two files and then randomly sample 2000 entries to be our test set

data = pd.concat([df1, df2]).reset_index(drop = True)

test = data.sample(2000)

data = data.drop(test.index).reset_index(drop=True)

test = test.reset_index(drop = True)
# Extract relevant columns into the train and test sets



x_train = data['headline']

y_train = data['is_sarcastic']



x_test = test['headline']

y_test = test['is_sarcastic']
# Generate vocabulary from the available dataset



tokenizer = Tokenizer()

tokenizer.fit_on_texts(pd.concat([x_train, x_test]))

word_index = tokenizer.word_index

vocab_size = len(word_index) + 1
# Tokenize our data



x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)
# Find the maximum length of our headlines



train_sizes = [len(l) for l in x_train]

test_sizes = [len(l) for l in x_test]

maxlen = np.max(train_sizes + test_sizes)
# Pad the data to get a uniform size



x_train_pad = sequence.pad_sequences(x_train, padding='post', maxlen=maxlen)

x_test_pad = sequence.pad_sequences(x_test, padding='post', maxlen=maxlen)
# Generate the model



model = Sequential() 

model.add(Embedding(vocab_size, 64, input_length=maxlen)) 

model.add(SimpleRNN(16, unroll=True))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adagrad', metrics = ['accuracy'])
# Train the model on our dataset



model.fit(

   x_train_pad, y_train, 

   batch_size = 128, 

   epochs = 3,

   validation_data = (x_test_pad, y_test)

)
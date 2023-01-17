# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



MAX_FEATURE = 100000

MAX_LEN = 100

EMBEDDING_DIM = 300

BATCH_SIZE = 512

EPOCHS = 2



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train_comments = train["comment_text"].astype(str)

train_targets = train["target"].astype("float32")

test_comments = test["comment_text"].astype(str)

test_ids = test["id"].astype(str)



tokenizer = Tokenizer(MAX_LEN)

tokenizer.fit_on_texts(train_comments)

sequences = tokenizer.texts_to_sequences(train_comments)

train_x = pad_sequences(sequences, MAX_LEN)



tokenizer.fit_on_texts(test_comments)

sequences = tokenizer.texts_to_sequences(test_comments)

test_x = pad_sequences(sequences, MAX_LEN)

train_y = np.where(train_targets >= 0.5, 1, 0)





# Any results you write to the current directory are saved as output.
from keras import models

from keras import layers

from keras import Input





input_tensor = Input(shape=(MAX_LEN,))

x = layers.Embedding(MAX_FEATURE, EMBEDDING_DIM, input_length=MAX_LEN)(input_tensor)

x = layers.Bidirectional(layers.LSTM(32 ,return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)

#attention = Attention.Attention(max_len)(x)

#attention = SeqSelfAttention(attention_activation='sigmoid')(x)

#attention = layers.GlobalMaxPooling1D()(attention)

x = layers.GlobalMaxPooling1D()(x)

#x = layers.concatenate([attention, x])

#x = layers.Dense(32, activation="relu")(x)

output_tensor = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(input_tensor, output_tensor)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

model.summary()
model.fit(train_x, train_y, epochs=EPOCHS, verbose=1, batch_size=BATCH_SIZE)
predictions = model.predict(test_x, batch_size=BATCH_SIZE, verbose=1)

predictions = predictions.ravel()



submission = pd.DataFrame.from_dict({

    'id': test_ids,

    'prediction': predictions

})



submission.to_csv('submission.csv', index=False)
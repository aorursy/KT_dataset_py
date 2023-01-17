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
import keras.backend as K

from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Embedding, LSTM, Input, Dropout, Flatten

from keras.models import Model, Sequential

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import ModelCheckpoint, EarlyStopping
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

submission_df = pd.read_csv('../input/sample_submission.csv')
train_df.shape, test_df.shape
# Fill NAN

train_df.fillna(' UNKNOWN ', inplace = True)

test_df.fillna(' UNKNOWN ', inplace = True)
tokenize = Tokenizer(num_words = 50000)
train_fit = tokenize.fit_on_texts(train_df['comment_text'].values)
train_text = tokenize.texts_to_sequences(train_df['comment_text'].values)

test_text = tokenize.texts_to_sequences(test_df['comment_text'].values)
X_train = pad_sequences(train_text, maxlen = 400)

X_test = pad_sequences(test_text, maxlen = 400)
y_train = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
print ("Training Data: {} {}".format(X_train.shape, y_train.shape))

print ("Testing Data: {}".format(X_test.shape))
def classification_model():

    K.clear_session()

    model = Sequential()

    model.add(Dense(32, input_shape = (400, ), activation = 'relu'))

    model.add(Dense(64, activation = 'relu'))

    model.add(Dense(128, activation = 'relu'))

    model.add(Dense(64, activation = 'relu'))

    model.add(Dense(32, activation = 'relu'))

    model.add(Dense(6, activation = 'sigmoid'))

    

    return model
model = classification_model()

model.summary()
file_path="weights_base.best.hdf5"



checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)



callback_list = [checkpoint, early]
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(X_train, y_train, 

          batch_size = 32, 

          epochs = 5, 

          validation_split = 0.1, 

          callbacks = callback_list)
model.load_weights('weights_base.best.hdf5')
model.evaluate(X_train, y_train)
pred = model.predict(X_test)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
submid = pd.DataFrame({'id': test_df["id"]})

submission = pd.concat([submid, pd.DataFrame(pred, columns = label_cols)], axis=1)

submission.to_csv('submission.csv', index=False)
submission.head()
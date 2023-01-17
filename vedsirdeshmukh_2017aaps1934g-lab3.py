import numpy as np 
import pandas as pd 
import os

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM, Input, ReLU, Conv1D, GlobalMaxPooling1D, Dropout, Activation
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import optimizers
 
import re
filelist = [ f for f in os.listdir() if f.endswith(".hdf5") ]
for f in filelist:
    os.remove(os.path.join(f))
data = pd.read_csv('/kaggle/input/nlpdata/nlp_train.csv')
pd.set_option('display.max_colwidth',-1)
data.head()

data.shape
for i in data.index:
    data['tweet'][i] = re.sub(r"http\S+", "", data['tweet'][i])
    data['tweet'][i] = re.sub(r"@\S+", "", data['tweet'][i])
    
data.head()
df=pd.read_csv("/kaggle/input/nlpdata/_nlp_test.csv",encoding="utf-8",index_col=0)
df = df.reset_index()
df = df.drop(['offensive_language'], axis=1)
num_words = 5000
tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')
print(data['tweet'][0])
tokenizer.fit_on_texts(data['tweet'].values)
X = tokenizer.texts_to_sequences(data['tweet'].values)
print(X[0])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

maxlen = max([len(x) for x in X])
X = pad_sequences(X, maxlen=maxlen)

print(word_index)
print("Padded Sequences: ")
print(X)
print(X[0])


test = df.iloc[:,0]
test = list(test)
print(test)

X_t = tokenizer.texts_to_sequences(test)

#word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))


X_t = pad_sequences(X_t, maxlen=maxlen)


print(X_t)
print(X_t[0])
maxlen
y = data['offensive_language']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
X_train
embed_dim = 100 
lstm_out = 128 
batch_size = 32

model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(num_words,
                    embed_dim,
                    input_length = maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(250,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(256))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('relu'))
model.summary()


opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
modelcheckpoint=ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#Training
#history = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size = batch_size, epochs = 20, callbacks=[earlystop,modelcheckpoint])
import matplotlib.pyplot as plt
# Plot loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# load weights into new model
model.load_weights("weights.04-0.35.hdf5")
print("Loaded model from disk")
loss = model.evaluate(X_test, y_test, batch_size = batch_size, callbacks=[earlystop])
loss
df=pd.read_csv("/kaggle/input/nlpdata/_nlp_test.csv",encoding="utf-8",index_col=0)
df = df.reset_index()
df = df.drop(['offensive_language'], axis=1)

df
df.shape



test = df.iloc[:,0]
test = list(test)
print(test)

#num_words = 5000
#tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   #lower=True,split=' ')
#print(data['tweet'][0])
#tokenizer.fit_on_texts(data['tweet'].values)
X_t = tokenizer.texts_to_sequences(test)

#word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))


X_t = pad_sequences(X_t, maxlen=maxlen)


print(X_t)
print(X_t[0])
y_pred = model.predict(X_t)
print(y_pred)
y_pred[:10]
df["offensive_language"] = y_pred
df
df.to_csv("submission.csv")
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode())
  payload = b64.decode()
  html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
  html = html.format(payload=payload,title=title,filename=filename)
  return HTML(html)
create_download_link(df)

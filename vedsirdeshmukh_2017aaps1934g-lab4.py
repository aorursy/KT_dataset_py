import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, Layer, Dropout, Lambda
from keras.optimizers import Adadelta
from keras.models import Model, Sequential
from keras.layers.embeddings import Embedding
from keras import backend as K
filelist = [ f for f in os.listdir() if f.endswith(".hdf5") ]
for f in filelist:
    os.remove(os.path.join(f))
data=pd.read_csv('../input/nnfl-lab-4/train.csv')
sentence1=data.Sentence1
sentence2=data.Sentence2
data.head()
data.shape
train_1 = data.iloc[:,1]
train_1 = list(train_1)

train_2 = data.iloc[:,2]
train_2 = list(train_2)

full_train = train_1 + train_2
print(full_train)
num_words = 5000
tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')

tokenizer.fit_on_texts(full_train)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print(word_index)
sen1_encoded = tokenizer.texts_to_sequences(data['Sentence1'].values)
print(sen1_encoded[0])

sen2_encoded = tokenizer.texts_to_sequences(data['Sentence2'].values)
print(sen2_encoded[0])

max_length_of_text = 60
sen1_encoded= pad_sequences(sen1_encoded, maxlen=max_length_of_text, padding='pre')
sen2_encoded= pad_sequences(sen2_encoded, maxlen=max_length_of_text, padding='pre')
print(sen1_encoded[:1])
print(sen2_encoded[:1])

y=data['Class']
y
x1_train=sen1_encoded
x2_train=sen2_encoded
y_train=y
from sklearn.model_selection import train_test_split

VALIDATION_RATIO = 0.2

x1_train, x1_val, \
x2_train, x2_val, \
y_train, y_val = \
    train_test_split(
        x1_train, x2_train, y_train, 
        test_size=VALIDATION_RATIO
)
print("Training dimensions...")
print(x1_train.shape)
print(x2_train.shape)

print("Validation dimensions...")
print(x1_val.shape)
print(x2_val.shape)

print("Output dimensions...")
print(y_train.shape)
print(y_val.shape)

NUM_CLASSES = 1


MAX_NUM_WORDS = 5000


MAX_SEQUENCE_LENGTH = max_length_of_text


NUM_EMBEDDING_DIM = 40


NUM_LSTM_UNITS = 256


BATCH_SIZE = 128


NUM_EPOCHS = 40
class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

from keras import optimizers
# Define the shared model
x = Sequential()
x.add(Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM))
# LSTM
x.add(LSTM(NUM_LSTM_UNITS))

shared_model = x

# The visible layer
left_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
right_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

# Pack it all up into a Manhattan Distance model
malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

opt = optimizers.Adadelta(clipnorm=2.5,learning_rate=1.0,rho=0.95)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
model.summary()
shared_model.summary()

from keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

earlystop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)

modelcheckpoint=ModelCheckpoint("weights.{epoch:02d}-{val_accuracy:.2f}.hdf5", monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

callbacks = [earlystop,modelcheckpoint,reduce_lr]

# Training Part
# history = model.fit([x1_train, x2_train], y_train,
                           #batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                           #validation_data=([x1_val, x2_val], y_val),callbacks=callbacks )
import matplotlib.pyplot as plt
# Plot accuracy
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
test_data=pd.read_csv('../input/nnfl-lab-4/test.csv')
sentence1=test_data.Sentence1
sentence2=test_data.Sentence2
test_data.head()
test_1 = test_data.iloc[:,1]
test_1 = list(test_1)

test_2 = test_data.iloc[:,2]
test_2 = list(test_2)

full_train = test_1 + test_2
print(full_train)
sen1_encoded = tokenizer.texts_to_sequences(test_data['Sentence1'].values)
print(sen1_encoded[0])
maxlen = 60
sen1_encoded = pad_sequences(sen1_encoded, maxlen=maxlen)
print("Padded Sequences: ")
print(sen1_encoded)
print(sen1_encoded[0])

sen2_encoded = tokenizer.texts_to_sequences(test_data['Sentence2'].values)
print(sen2_encoded[0])
maxlen = 60
sen2_encoded = pad_sequences(sen2_encoded, maxlen=maxlen)
print("Padded Sequences: ")
print(sen2_encoded)
print(sen2_encoded[0])
x1_test=sen1_encoded
x2_test=sen2_encoded
# load weights into new model
model.load_weights("weights.09-0.68.hdf5")
print("Loaded model from disk")
predictions = model.predict([x1_test, x2_test])
predictions[:10]
for n in range(len(predictions)):
    if predictions[n] > 0.5:
        predictions[n]=1
    else:
        predictions[n]=0
predictions=predictions.astype(int)
predictions[:10]
test_data = test_data.drop(['Sentence1','Sentence2'], axis=1)
test_data['Class'] = predictions
test_data
from IPython.display import HTML
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode())
  payload = b64.decode()
  html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
  html = html.format(payload=payload,title=title,filename=filename)
  return HTML(html)
create_download_link(test_data)
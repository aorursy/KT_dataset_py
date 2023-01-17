from __future__ import print_function

import os

import re

import tqdm

import string

import pandas as pd

import numpy as np

import keras

from keras.layers import Flatten
#train data

df_train_txt = pd.read_csv('../input/training_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])

df_train_var = pd.read_csv('../input/training_variants')

#test data

df_test_txt = pd.read_csv('../input/test_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])

df_test_var = pd.read_csv('../input/test_variants')
train = df_train_var.merge(df_train_txt, how="inner", left_on="ID", right_on="ID")

train.head()
test_x = df_test_var.merge(df_test_txt, how="inner", left_on="ID", right_on="ID")

test_x.head()
train_y = train['Class'].values

train_x = train.drop('Class', axis=1)

train_size=len(train_x)

print('Number of training variants: %d' % (train_size))
test_size=len(test_x)

print('Number of test variants: %d' % (test_size))

# number of test data : 5668
test_index = test_x['ID'].values

all_data = np.concatenate((train_x, test_x), axis=0)

all_data = pd.DataFrame(all_data)

all_data.columns = ["ID", "Gene", "Variation", "Text"]
all_data.head()
from nltk.corpus import stopwords

from gensim.models.doc2vec import LabeledSentence

from gensim import utils
def constructLabeledSentences(data):

    sentences=[]

    for index, row in data.iteritems():

        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))

    return sentences



def textClean(text):

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = text.lower().split()

    stops = set(stopwords.words("english"))

    text = [w for w in text if not w in stops]    

    text = " ".join(text)

    return(text)

    

def cleanup(text):

    text = textClean(text)

    text= text.translate(str.maketrans("","", string.punctuation))

    return text



allText = all_data['Text'].apply(cleanup)

sentences = constructLabeledSentences(allText)

allText.head()
from gensim.models import Doc2Vec
Text_INPUT_DIM=300





text_model=None

filename='docEmbeddings_5_clean.d2v'

if os.path.isfile(filename):

    text_model = Doc2Vec.load(filename)

else:

    text_model = Doc2Vec(min_count=1, window=5, size=Text_INPUT_DIM, sample=1e-4, negative=5, workers=4, iter=5,seed=1)

    text_model.build_vocab(sentences)

    text_model.train(sentences, total_examples=text_model.corpus_count, epochs=text_model.iter)

    text_model.save(filename)
text_train_arrays = np.zeros((train_size, Text_INPUT_DIM))

text_test_arrays = np.zeros((test_size, Text_INPUT_DIM))



for i in range(train_size):

    text_train_arrays[i] = text_model.docvecs['Text_'+str(i)]



j=0

for i in range(train_size,train_size+test_size):

    text_test_arrays[j] = text_model.docvecs['Text_'+str(i)]

    j=j+1
from sklearn.decomposition import TruncatedSVD

Gene_INPUT_DIM=25



#Gene and Variation Featurizer.

svd = TruncatedSVD(n_components=25, n_iter=Gene_INPUT_DIM, random_state=12)



one_hot_gene = pd.get_dummies(all_data['Gene'])

truncated_one_hot_gene = svd.fit_transform(one_hot_gene.values)



one_hot_variation = pd.get_dummies(all_data['Variation'])

truncated_one_hot_variation = svd.fit_transform(one_hot_variation.values)
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder



#One hot vector encoding of classes

label_encoder = LabelEncoder()

label_encoder.fit(train_y)

encoded_y = np_utils.to_categorical((label_encoder.transform(train_y)))
##Considering all features, i.e. Gene, Variation and Text.



#Merge Input Features

train_set=np.hstack((truncated_one_hot_gene[:train_size],truncated_one_hot_variation[:train_size],text_train_arrays))

test_set=np.hstack((truncated_one_hot_gene[train_size:],truncated_one_hot_variation[train_size:],text_test_arrays))

train_set = np.reshape(train_set, (train_set.shape[0],1,train_set.shape[1]))
train_set.shape
from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector

from keras.optimizers import SGD, Adam, RMSprop, Adamax



def baseline_model(r,c):

    #----------------Old Start-----------------   

    #     model = Sequential()

    #     model.add(Dense(256, input_dim=Text_INPUT_DIM+Gene_INPUT_DIM*2, init='normal', activation='relu'))

    #     model.add(Dropout(0.1))

    #     model.add(Dense(256, init='normal', activation='relu'))

    #     model.add(Dropout(0.08))

    #     model.add(Dense(80, init='normal', activation='relu'))

    #     model.add(Dense(9, init='normal', activation="softmax"))

    #     adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    #     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #     return model 

    #----------------Old end-----------------



    #------------------------------Modified Start--------------------------

    lstm_out = 350

    model = Sequential()

    model.add(LSTM(lstm_out,return_sequences=True, dropout=0.6, input_shape=(1, c)))

#     model.add(LSTM(lstm_out,return_sequences=True, dropout=0.6, input_shape=(1, c)))

    model.add(Dropout(1.8))

    model.add(Dense(10, init='normal', activation='relu'))

    model.add(Dense(10, init='normal', activation='relu'))

#     model.add(Dropout(0.08))

#     model.add(Flatten())

    

    model.add(Dense(9, init='normal', activation="softmax"))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    return model 

    #------------------------------Modified End--------------------------

##sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  



##rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

##adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)



# '''

# model = Sequential()

# model.add(Embedding(2000, 128, input_length = X.shape[1]))

# model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))

# model.add(Dense(9,activation='softmax'))

# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])

# '''

# model = baseline_model(train_set.shape[1],train_set.shape[2])

model = baseline_model(train_set.shape[0],train_set.shape[2])

model.summary()
encoded_y.shape
encoded_y = np.reshape(encoded_y, (encoded_y.shape[0],1,encoded_y.shape[1]))
# train_set = np.reshape(train_set, (train_set.shape[0], 1, train_set.shape[1]))

# X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

train_set.shape

estimator=model.fit(train_set, encoded_y, validation_split=0.2, epochs=2, batch_size=32)
import matplotlib.pyplot as plt



# summarize history for accuracy

plt.plot(estimator.history['acc'])

plt.plot(estimator.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(estimator.history['loss'])

plt.plot(estimator.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper left')

plt.show()
print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % (100*estimator.history['acc'][-1], 100*estimator.history['val_acc'][-1]))
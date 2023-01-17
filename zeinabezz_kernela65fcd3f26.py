#save weight embedding in glove + multichannel cnnlstm +validation

from pickle import load

from keras.layers import Layer

from numpy import array

from keras.preprocessing.text import Tokenizer

from gensim.models import Word2Vec

from keras.preprocessing.sequence import pad_sequences

from numpy import asarray

from keras.utils.vis_utils import plot_model

from keras.models import Model

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from scipy.stats import pearsonr

from sklearn.metrics import mean_squared_error

import numpy as np

from keras.layers import Input

from numpy import zeros

from keras.layers import Dense

from keras.layers import Flatten

from gensim.models import Word2Vec

from keras.layers import Dropout

from keras.layers import Embedding

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.layers.merge import concatenate

import tensorflow as tf

import numpy

from sklearn.model_selection import KFold

import pandas as pd

from nltk.tokenize import sent_tokenize

from keras import backend as k

from keras import optimizers

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer 

from tensorflow.python.keras import backend as K

import keras

from keras.layers import LSTM

from keras import backend as K

from numpy import asarray

from keras.callbacks import LearningRateScheduler

import numpy as np

import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping

    



def correlation_coefficient(y_true, y_pred):

    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]





def construct_embedding():

    # load the whole embedding into memory

    embeddings_index = dict()

    f = open('../input/embedding/word_embedding50d.txt')

    for line in f:

        values = line.split()

        word = values[0]

        coefs = asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs

    f.close()

    print('Loaded %s word vectors.' % len(embeddings_index))

    return embeddings_index









# define the model

def define_model(length, vocab_size, embedding_matrix):

	# channel 1

    inputs1 = Input(shape=(length,))

    embedding1 = Embedding(vocab_size, 50, weights=[embedding_matrix],trainable=False)(inputs1)

    conv1 = Conv1D(filters=350, kernel_size=3, activation='relu')(embedding1)

    drop1 = Dropout(0.2)(conv1)

    nor11=keras.layers.BatchNormalization()(drop1)



    pool1 = MaxPooling1D(pool_size=5)(nor11)

    pdrop1 = Dropout(0.2)(pool1)

    nor12=keras.layers.BatchNormalization()(pdrop1)



    ls1=LSTM(200)(nor12)

    ldrop1 = Dropout(0.2)(ls1)

    lnor1=keras.layers.BatchNormalization()(ldrop1)



	# channel 2

    inputs2 = Input(shape=(length,))

    embedding2 = Embedding(vocab_size, 50, weights=[embedding_matrix],trainable=False)(inputs2)

    conv2 = Conv1D(filters=350, kernel_size=4, activation='relu')(embedding2)

    drop2 = Dropout(0.2)(conv2)

    nor21=keras.layers.BatchNormalization()(drop2)



    pool2 = MaxPooling1D(pool_size=5)(nor21)

    pdrop2 = Dropout(0.2)(pool2) 

    nor22=keras.layers.BatchNormalization()(pdrop2)

    

    ls2=LSTM(200)(nor22)

    ldrop2 = Dropout(0.2)(ls2)

    lnor2=keras.layers.BatchNormalization()(ldrop2)



    

	# channel 3

    inputs3 = Input(shape=(length,))

    embedding3 = Embedding(vocab_size, 50, weights=[embedding_matrix],trainable=False)(inputs3)

    conv3 = Conv1D(filters=350, kernel_size=5, activation='relu')(embedding3)

    drop3 = Dropout(0.2)(conv3)

    nor31=keras.layers.BatchNormalization()(drop3)



    pool3 = MaxPooling1D(pool_size=5)(nor31)

    pdrop3 = Dropout(0.2)(pool3) 

    nor32=keras.layers.BatchNormalization()(pdrop3)



    

    ls3=LSTM(250)(nor32)

    ldrop3 = Dropout(0.2)(ls3)

    lnor3=keras.layers.BatchNormalization()(ldrop3)

    



	# merge

    merged=concatenate([lnor1, lnor2, lnor3])

    # interpretation

    dense1 = Dense(100, activation='elu')(merged)

    nor4=keras.layers.BatchNormalization()(dense1)

    #dense2 = Dense(50, activation='elu')(nor4)

    #nor5=keras.layers.BatchNormalization()(dense2)



    outputs = Dense(1, activation='elu')(nor4)

    noroutputs=keras.layers.BatchNormalization()(outputs)

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=noroutputs)

    # compile

    model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.003), metrics=[correlation_coefficient, 'accuracy'])

    K.get_session().run(tf.local_variables_initializer())



    # summarize 

    print(model.summary())

    plot_model(model, show_shapes=True, to_file='multichannel.png')



    return model



# load a clean dataset

def load_dataset(filename):

    return load(open(filename, 'rb'))



#preprocessing text

def preprocess(lines):

	#print(lines)      

	ps = PorterStemmer() 

	for i in range(len(lines)):

		tokens = lines[i].split() 

        # filter out stop words then stem the remaining words

		stop_words = set(stopwords.words('english'))    

		tokens = [ps.stem(w) for w in tokens if not w in stop_words]    

		lines[i]=' '.join(tokens)  

	#print('lines: ')

	#print(lines)

	return lines





# encode a list of lines

def encode_text(tokenizer, lines, length):  

	# integer encode

	encoded = tokenizer.texts_to_sequences(lines)

	# pad encoded sequences

	padded = pad_sequences(encoded, maxlen=length, padding='post')

	return padded





# fit a tokenizer

def create_tokenizer(lines):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer



# calculate the maximum document length

def max_length(lines):

    return max([len(s.split()) for s in lines])



# encode a list of lines

def encode_text(tokenizer, lines, length):

	# integer encode

    encoded = tokenizer.texts_to_sequences(lines)

	# pad encoded sequences

    padded = pad_sequences(encoded, maxlen=length, padding='post')

    return padded



def embed (vocab_size, embeddings_index, t):

    embedding_matrix = zeros((vocab_size, 50))

    for word, i in t.word_index.items():

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

    return embedding_matrix





# define training data

f = pd.read_csv('../input/satexasdataset/texasDatasetForHAN.csv', encoding='ISO-8859-1')

ftest = pd.read_csv('../input/satexasdataset/testtexasDatasetForHAN.csv', encoding='ISO-8859-1')



train=[]

test=[]



#print(f.read())

data_train = pd.DataFrame(data=f) 



for i in range(data_train.shape[0]):

    train.append(data_train.manswer[i]+ ' ' +data_train.sanswer[i])



trainLabels=data_train.score

Lines=pd.DataFrame(train, columns=['train'])

trainLines=Lines.train

trainLines=preprocess(trainLines)



data_test = pd.DataFrame(data=ftest) 

for i in range(data_test .shape[0]):

    test.append(data_test.manswer[i] + ' '+data_test.sanswer[i])

    

testLabels=data_test.score

tLines=pd.DataFrame(test, columns=['test'])

testLines=tLines.test

testLines=preprocess(testLines)



mergedLines = [trainLines , testLines]

allmerged = pd.concat(mergedLines)



# create tokenizer

tokenizer = create_tokenizer(allmerged.str.lower())





# calculate max document length

length = max_length(allmerged)



# calculate vocabulary size

vocab_size = len(tokenizer.word_index) + 1





print('Max answerlength: %d' % length)

print('Vocabulary size: %d' % vocab_size)



# encode data

alldataX = encode_text(tokenizer, allmerged, length)





s=(trainLines.size)

trainX=alldataX[0:s]

testX=alldataX[s:]



    

    

print(trainX.shape,  testX.shape)

embeddings_index=construct_embedding()

embedding_matrix=embed (vocab_size, embeddings_index, tokenizer)



kfold = KFold(2, True, 1)

traincvlossscores=[]

traincvscores=[]

cvlossscores=[]

cvscores=[]

histories=[]

val_histories=[]

cor_histories=[]

val_cor_histories=[]





	# enumerate splits

for train, test in kfold.split(trainX):



	# define model

	model = define_model(length, vocab_size, embedding_matrix)

		

	mcp=keras.callbacks.ModelCheckpoint('bestweights.hdf5', monitor='val_loss', save_best_only=True, verbose=1)

	callbacks_list = [mcp]



	# fit model

	model.fit([trainX[train], trainX[train], trainX[train]], array(trainLabels[train]), epochs=3, batch_size=80, 

                      callbacks = callbacks_list,  validation_data=([trainX[test], trainX[test], trainX[test]],trainLabels[test]))



	# save the model

	model.save('model.h5')

    

    # evaluate model on test dataset dataset

loss, cor, acc = model.evaluate([testX,testX,testX],array(testLabels), verbose=0)

print('Test Correlation: %f' % (cor*100))

print('Test loss: %f' % (loss*100))

print('Test acc: %f' % (acc*100))

#load weight embedding in glove + multichannel cnnlstm +validation

from pickle import load

from keras.layers import Layer

from numpy import array

from keras.preprocessing.text import Tokenizer

from gensim.models import Word2Vec

from keras.preprocessing.sequence import pad_sequences

from numpy import asarray

from keras.utils.vis_utils import plot_model

from keras.models import Model

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from scipy.stats import pearsonr

from sklearn.metrics import mean_squared_error

import numpy as np

from keras.layers import Input

from numpy import zeros

from keras.layers import Dense

from keras.layers import Flatten

from gensim.models import Word2Vec

from keras.layers import Dropout

from keras.layers import Embedding

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.layers.merge import concatenate

import tensorflow as tf

import numpy

from sklearn.model_selection import KFold

import pandas as pd

from nltk.tokenize import sent_tokenize

from keras import backend as k

from keras import optimizers

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer 

from tensorflow.python.keras import backend as K

import keras

from keras.layers import LSTM

from keras import backend as K

from numpy import asarray

from keras.callbacks import LearningRateScheduler

import numpy as np

import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping





def correlation_coefficient(y_true, y_pred):

    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]





def construct_embedding():

    # load the whole embedding into memory

    embeddings_index = dict()

    f = open('../input/embedding/word_embedding50d.txt')

    for line in f:

        values = line.split()

        word = values[0]

        coefs = asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs

    f.close()

    print('Loaded %s word vectors.' % len(embeddings_index))

    return embeddings_index









# define the model

def define_model(length, vocab_size, embedding_matrix):

	# channel 1

    inputs1 = Input(shape=(length,))

    embedding1 = Embedding(vocab_size, 50, weights=[embedding_matrix],trainable=False)(inputs1)

    conv1 = Conv1D(filters=350, kernel_size=3, activation='relu')(embedding1)

    drop1 = Dropout(0.2)(conv1)

    nor11=keras.layers.BatchNormalization()(drop1)



    pool1 = MaxPooling1D(pool_size=5)(nor11)

    pdrop1 = Dropout(0.2)(pool1)

    nor12=keras.layers.BatchNormalization()(pdrop1)



    ls1=LSTM(200)(nor12)

    ldrop1 = Dropout(0.2)(ls1)

    lnor1=keras.layers.BatchNormalization()(ldrop1)



	# channel 2

    inputs2 = Input(shape=(length,))

    embedding2 = Embedding(vocab_size, 50, weights=[embedding_matrix],trainable=False)(inputs2)

    conv2 = Conv1D(filters=350, kernel_size=4, activation='relu')(embedding2)

    drop2 = Dropout(0.2)(conv2)

    nor21=keras.layers.BatchNormalization()(drop2)



    pool2 = MaxPooling1D(pool_size=5)(nor21)

    pdrop2 = Dropout(0.2)(pool2) 

    nor22=keras.layers.BatchNormalization()(pdrop2)

    

    ls2=LSTM(200)(nor22)

    ldrop2 = Dropout(0.2)(ls2)

    lnor2=keras.layers.BatchNormalization()(ldrop2)



    

	# channel 3

    inputs3 = Input(shape=(length,))

    embedding3 = Embedding(vocab_size, 50, weights=[embedding_matrix],trainable=False)(inputs3)

    conv3 = Conv1D(filters=350, kernel_size=5, activation='relu')(embedding3)

    drop3 = Dropout(0.2)(conv3)

    nor31=keras.layers.BatchNormalization()(drop3)



    pool3 = MaxPooling1D(pool_size=5)(nor31)

    pdrop3 = Dropout(0.2)(pool3) 

    nor32=keras.layers.BatchNormalization()(pdrop3)



    

    ls3=LSTM(250)(nor32)

    ldrop3 = Dropout(0.2)(ls3)

    lnor3=keras.layers.BatchNormalization()(ldrop3)

    



	# merge

    merged=concatenate([lnor1, lnor2, lnor3])

    # interpretation

    dense1 = Dense(100, activation='elu')(merged)

    nor4=keras.layers.BatchNormalization()(dense1)

 



    outputs = Dense(1, activation='elu')(nor4)

    noroutputs=keras.layers.BatchNormalization()(outputs)

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=noroutputs)

    model.load_weights("../input/bestweight/bestweights.hdf5")

    # compile

    model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.003), metrics=[correlation_coefficient, 'accuracy'])

    K.get_session().run(tf.local_variables_initializer())



    # summarize 

    print(model.summary())

    plot_model(model, show_shapes=True, to_file='multichannel.png')



    return model



# load a clean dataset

def load_dataset(filename):

    return load(open(filename, 'rb'))



#preprocessing text

def preprocess(lines):

	#print(lines)      

	ps = PorterStemmer() 

	for i in range(len(lines)):

		tokens = lines[i].split() 

        # filter out stop words then stem the remaining words

		stop_words = set(stopwords.words('english'))    

		tokens = [ps.stem(w) for w in tokens if not w in stop_words]    

		lines[i]=' '.join(tokens)  

	#print('lines: ')

	#print(lines)

	return lines



    



# encode a list of lines

def encode_text(tokenizer, lines, length):  

	# integer encode

	encoded = tokenizer.texts_to_sequences(lines)

	# pad encoded sequences

	padded = pad_sequences(encoded, maxlen=length, padding='post')

	return padded





# fit a tokenizer

def create_tokenizer(lines):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer



# calculate the maximum document length

def max_length(lines):

    return max([len(s.split()) for s in lines])



# encode a list of lines

def encode_text(tokenizer, lines, length):

	# integer encode

    encoded = tokenizer.texts_to_sequences(lines)

	# pad encoded sequences

    padded = pad_sequences(encoded, maxlen=length, padding='post')

    return padded



def embed (vocab_size, embeddings_index, t):

    embedding_matrix = zeros((vocab_size, 50))

    for word, i in t.word_index.items():

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

    return embedding_matrix





# define training data

f = pd.read_csv('../input/satexasdataset/texasDatasetForHAN.csv', encoding='ISO-8859-1')

ftest = pd.read_csv('../input/satexasdataset/testtexasDatasetForHAN.csv', encoding='ISO-8859-1')



train=[]

test=[]



#print(f.read())

data_train = pd.DataFrame(data=f) 



for i in range(data_train.shape[0]):

    train.append(data_train.manswer[i]+ ' ' +data_train.sanswer[i])



trainLabels=data_train.score

Lines=pd.DataFrame(train, columns=['train'])

trainLines=Lines.train

trainLines=preprocess(trainLines)



data_test = pd.DataFrame(data=ftest) 

for i in range(data_test .shape[0]):

    test.append(data_test.manswer[i] + ' '+data_test.sanswer[i])

    

testLabels=data_test.score

tLines=pd.DataFrame(test, columns=['test'])

testLines=tLines.test

testLines=preprocess(testLines)



mergedLines = [trainLines , testLines]

allmerged = pd.concat(mergedLines)



# create tokenizer

tokenizer = create_tokenizer(allmerged.str.lower())





# calculate max document length

length = max_length(allmerged)



# calculate vocabulary size

vocab_size = len(tokenizer.word_index) + 1





print('Max answerlength: %d' % length)

print('Vocabulary size: %d' % vocab_size)



# encode data

alldataX = encode_text(tokenizer, allmerged, length)





s=(trainLines.size)

trainX=alldataX[0:s]

testX=alldataX[s:]



    

    

print(trainX.shape,  testX.shape)



embeddings_index=construct_embedding()

embedding_matrix=embed (vocab_size, embeddings_index, tokenizer)





traincvlossscores=[]

traincvscores=[]

cvlossscores=[]

cvscores=[]

histories=[]

val_histories=[]

cor_histories=[]

val_cor_histories=[]





	# enumerate splits



	# define model

model = define_model(length, vocab_size, embedding_matrix)





# evaluate model on test dataset dataset

loss, cor, acc = model.evaluate([testX,testX,testX],array(testLabels), verbose=0)

print('Test Correlation: %f' % (cor*100))

print('Test loss: %f' % (loss*100))

print('Test acc: %f' % (acc*100))

#load weight embedding in glove + multichannel cnnlstm +validation

from pickle import load

from keras.layers import Layer

from numpy import array

from keras.preprocessing.text import Tokenizer

from gensim.models import Word2Vec

from keras.preprocessing.sequence import pad_sequences

from numpy import asarray

from keras.utils.vis_utils import plot_model

from keras.models import Model

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from scipy.stats import pearsonr

from sklearn.metrics import mean_squared_error

import numpy as np

from keras.layers import Input

from numpy import zeros

from keras.layers import Dense

from keras.layers import Flatten

from gensim.models import Word2Vec

from keras.layers import Dropout

from keras.layers import Embedding

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.layers.merge import concatenate

import tensorflow as tf

import numpy

from sklearn.model_selection import KFold

import pandas as pd

from nltk.tokenize import sent_tokenize

from keras import backend as k

from keras import optimizers

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer 

from tensorflow.python.keras import backend as K

import keras

from keras.layers import LSTM

from keras import backend as K

from numpy import asarray

from keras.callbacks import LearningRateScheduler

import numpy as np

import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping





def correlation_coefficient(y_true, y_pred):

    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]





def construct_embedding():

    # load the whole embedding into memory

    embeddings_index = dict()

    f = open('../input/embedding/word_embedding50d.txt')

    for line in f:

        values = line.split()

        word = values[0]

        coefs = asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs

    f.close()

    print('Loaded %s word vectors.' % len(embeddings_index))

    return embeddings_index









# define the model

def define_model(length, vocab_size, embedding_matrix):

	# channel 1

    inputs1 = Input(shape=(length,))

    embedding1 = Embedding(vocab_size, 50, weights=[embedding_matrix],trainable=False)(inputs1)

    conv1 = Conv1D(filters=350, kernel_size=3, activation='relu')(embedding1)

    drop1 = Dropout(0.2)(conv1)

    nor11=keras.layers.BatchNormalization()(drop1)



    pool1 = MaxPooling1D(pool_size=5)(nor11)

    pdrop1 = Dropout(0.2)(pool1)

    nor12=keras.layers.BatchNormalization()(pdrop1)



    ls1=LSTM(200)(nor12)

    ldrop1 = Dropout(0.2)(ls1)

    lnor1=keras.layers.BatchNormalization()(ldrop1)



	# channel 2

    inputs2 = Input(shape=(length,))

    embedding2 = Embedding(vocab_size, 50, weights=[embedding_matrix],trainable=False)(inputs2)

    conv2 = Conv1D(filters=350, kernel_size=4, activation='relu')(embedding2)

    drop2 = Dropout(0.2)(conv2)

    nor21=keras.layers.BatchNormalization()(drop2)



    pool2 = MaxPooling1D(pool_size=5)(nor21)

    pdrop2 = Dropout(0.2)(pool2) 

    nor22=keras.layers.BatchNormalization()(pdrop2)

    

    ls2=LSTM(200)(nor22)

    ldrop2 = Dropout(0.2)(ls2)

    lnor2=keras.layers.BatchNormalization()(ldrop2)



    

	# channel 3

    inputs3 = Input(shape=(length,))

    embedding3 = Embedding(vocab_size, 50, weights=[embedding_matrix],trainable=False)(inputs3)

    conv3 = Conv1D(filters=350, kernel_size=5, activation='relu')(embedding3)

    drop3 = Dropout(0.2)(conv3)

    nor31=keras.layers.BatchNormalization()(drop3)



    pool3 = MaxPooling1D(pool_size=5)(nor31)

    pdrop3 = Dropout(0.2)(pool3) 

    nor32=keras.layers.BatchNormalization()(pdrop3)



    

    ls3=LSTM(250)(nor32)

    ldrop3 = Dropout(0.2)(ls3)

    lnor3=keras.layers.BatchNormalization()(ldrop3)

    



	# merge

    merged=concatenate([lnor1, lnor2, lnor3])

    # interpretation

    dense1 = Dense(100, activation='elu')(merged)

    nor4=keras.layers.BatchNormalization()(dense1)

 



    outputs = Dense(1, activation='elu')(nor4)

    noroutputs=keras.layers.BatchNormalization()(outputs)

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=noroutputs)

    model.load_weights("../input/bestweight/bestweights.hdf5")

    # compile

    model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.003), metrics=[correlation_coefficient, 'accuracy'])

    K.get_session().run(tf.local_variables_initializer())



    # summarize 

    print(model.summary())

    plot_model(model, show_shapes=True, to_file='multichannel.png')



    return model



# load a clean dataset

def load_dataset(filename):

    return load(open(filename, 'rb'))



#preprocessing text

def preprocess(lines):

	#print(lines)      

	ps = PorterStemmer() 

	for i in range(len(lines)):

		tokens = lines[i].split() 

        # filter out stop words then stem the remaining words

		stop_words = set(stopwords.words('english'))    

		tokens = [ps.stem(w) for w in tokens if not w in stop_words]    

		lines[i]=' '.join(tokens)  

	#print('lines: ')

	#print(lines)

	return lines



    



# encode a list of lines

def encode_text(tokenizer, lines, length):  

	# integer encode

	encoded = tokenizer.texts_to_sequences(lines)

	# pad encoded sequences

	padded = pad_sequences(encoded, maxlen=length, padding='post')

	return padded





# fit a tokenizer

def create_tokenizer(lines):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer



# calculate the maximum document length

def max_length(lines):

    return max([len(s.split()) for s in lines])



# encode a list of lines

def encode_text(tokenizer, lines, length):

	# integer encode

    encoded = tokenizer.texts_to_sequences(lines)

	# pad encoded sequences

    padded = pad_sequences(encoded, maxlen=length, padding='post')

    return padded



def embed (vocab_size, embeddings_index, t):

    embedding_matrix = zeros((vocab_size, 50))

    for word, i in t.word_index.items():

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

    return embedding_matrix



def cal(pred, actual):

	pred=pd.DataFrame(pred,  columns=['predict'])

	yy=pred.predict

	l=np.array(actual, dtype=pd.Series)

	l=l.tolist()

	corr, _ = pearsonr(yy, l)

	mean=mean_squared_error(yy, l)

	return corr, mean

 





# define training data

f = pd.read_csv('../input/satexasdataset/texasDatasetForHAN.csv', encoding='ISO-8859-1')

ftest = pd.read_csv('../input/satexasdataset/testtexasDatasetForHAN.csv', encoding='ISO-8859-1')



train=[]

test=[]



#print(f.read())

data_train = pd.DataFrame(data=f) 



for i in range(data_train.shape[0]):

    train.append(data_train.manswer[i]+ ' ' +data_train.sanswer[i])



trainLabels=data_train.score

Lines=pd.DataFrame(train, columns=['train'])

trainLines=Lines.train

trainLines=preprocess(trainLines)



data_test = pd.DataFrame(data=ftest) 

for i in range(data_test .shape[0]):

    test.append(data_test.manswer[i] + ' '+data_test.sanswer[i])

    

#testLabels=data_test.score

testLabels = data_test.score.astype(np.float32)

tLines=pd.DataFrame(test, columns=['test'])

testLines=tLines.test

testLines=preprocess(testLines)



mergedLines = [trainLines , testLines]

allmerged = pd.concat(mergedLines)



# create tokenizer

tokenizer = create_tokenizer(allmerged.str.lower())





# calculate max document length

length = max_length(allmerged)



# calculate vocabulary size

vocab_size = len(tokenizer.word_index) + 1





print('Max answerlength: %d' % length)

print('Vocabulary size: %d' % vocab_size)



# encode data

alldataX = encode_text(tokenizer, allmerged, length)





s=(trainLines.size)

trainX=alldataX[0:s]

testX=alldataX[s:]



    

    

print(trainX.shape,  testX.shape)



embeddings_index=construct_embedding()

embedding_matrix=embed (vocab_size, embeddings_index, tokenizer)





traincvlossscores=[]

traincvscores=[]

cvlossscores=[]

cvscores=[]

histories=[]

val_histories=[]

cor_histories=[]

val_cor_histories=[]





	# enumerate splits



	# define model

model = define_model(length, vocab_size, embedding_matrix)





ynew = model.predict([testX,testX,testX])

print('Test Correlation1: ', cal((ynew), (testLabels)))



corr=correlation_coefficient(ynew, testLabels)

K.get_session().run(tf.local_variables_initializer())

print('Test Correlation2: ', K.get_session().run(corr))

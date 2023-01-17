import pandas as pd

import numpy as np

import os



# Dataset Loading

'''

folder = 'aclImdb'

labels = {'pos': 1, 'neg': 0}

df = pd.DataFrame()



for f in ('test', 'train'):    

    for l in ('pos', 'neg'):

        path = os.path.join(folder, f, l)

        for file in os.listdir (path) :

            with open(os.path.join(path, file),'r', encoding='utf-8') as infile:

                txt = infile.read()

            df = df.append([[txt, labels[l]]],ignore_index = True)

            

df.columns = ['review', 'sentiment']

df.to_csv('movie_data_lstm.csv', index = False, encoding = 'utf-8')

'''

df = pd.read_csv('../input/imdb00/movie_data_lstm.csv', encoding = 'utf-8')

df.head(10)
# Data Preprocessing



import string 

import nltk

import multiprocessing

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

EMB_DIM = 300



review_lines = list()

lines = df['review'].values.tolist()

max_len = 0

#vocab_size = []

for line in lines:

    line = line.replace('<br />', '')

    tokens = word_tokenize(line)

    # Convert to lower case

    tokens = [w.lower() for w in tokens]

    #print(tokens)

    # Remove punctuation from each word

    table = str.maketrans('', '', string.punctuation)

    stripped = [w.translate(table) for w in tokens]

    # Remove remaining tokens that are not alphabetic

    words = [word for word in stripped if word.isalpha()]

    #print(words)

    #print(type(words))

    #Filter out stop words.

    #stop_words = set(stopwords.words('english'))

    #words = [w for w in words if not w in stop_words]

    #print(words)

    if(len(words) > max_len):

        max_len = len(words)

        

    review_lines.append(words)

    

    '''

    for w in words:

        if w not in vocab_size:

            vocab_size.append(w)

            

print(len(vocab_size)) 

'''

print(max_len)

len(review_lines)
# Word2Vec Embedding Training



import gensim



model = gensim.models.Word2Vec(sentences = review_lines, 

                               size = EMB_DIM, 

                               window = 5, 

                               workers = multiprocessing.cpu_count(), 

                               min_count = 1)



words = list(model.wv.vocab)

print('Vocabulary Size: %d' % len(words))
# Random test for the Trained Embeddings



model.wv.most_similar('horrible')
# Saving the  generated Embeddings in a text file for further use



filename = 'imdb_embedding_word2vec.txt'

model.wv.save_word2vec_format(filename, binary = False)
# Loading the generated Embeddings



import os



embeddings_index = {}

f = open(os.path.join('', 'imdb_embedding_word2vec.txt'), encoding = 'utf-8')

for line in f:

    values = line.split()

    word = values[0]

    foo = np.asarray(values[1:])

    embeddings_index[word] = foo

    

f.close()
# Extension of Data Preprocessing



from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences



'''

The documents are integer encoded prior to passing them to the Embedding layer. 

The integer maps to the index of a specific vector in the Embedding layer. 

Therefore, it is important to lay the vectors out in the Embedding layer such that the 

encoded words map to the correct vector.

'''



# vectorize the text samples into a 2d integer tensor

tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(review_lines)

sequences = tokenizer_obj.texts_to_sequences(review_lines)





word_index = tokenizer_obj.word_index

print('Found %s unique tokens' % len(word_index))



review_pad = pad_sequences(sequences, maxlen = max_len)

print(review_pad.shape)

sentiment = df['sentiment'].values

print('Shape of review tensor: ', review_pad.shape)

print('Shape of sentiment tensor: ', sentiment.shape)
'''

Map embeddings from the loaded word2vec model for each word to the tokenizer_obj.word_index 

vocabulary and creating a matrix of word vectors.

'''



num_words = len(word_index) + 1

embedding_matrix  = np.zeros((num_words, EMB_DIM))



for word, i in word_index.items():

    if i > num_words:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in the embedding matrix will all be zeros

        embedding_matrix[i] = embedding_vector

        

print(num_words)
# Model Creation



from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional

from keras.layers.embeddings import Embedding

from keras.initializers import Constant

from keras.utils.vis_utils import plot_model



model = Sequential()

embedding_layer = Embedding(num_words,

                            EMB_DIM, 

                            embeddings_initializer = Constant(embedding_matrix), 

                            input_length = max_len,

                            trainable = False)



model.add(embedding_layer)

model.add(Bidirectional(LSTM(200, activation = 'tanh', 

                             recurrent_activation = 'hard_sigmoid', 

                             use_bias = True, 

                             kernel_initializer = 'glorot_uniform', 

                             recurrent_initializer = 'orthogonal', 

                             bias_initializer = 'zeros', 

                             unit_forget_bias = True, 

                             kernel_regularizer = None, 

                             recurrent_regularizer = None, 

                             bias_regularizer = None, 

                             activity_regularizer = None, 

                             kernel_constraint = None, 

                             recurrent_constraint = None, 

                             bias_constraint = None, 

                             dropout = 0.0, 

                             recurrent_dropout = 0.0, 

                             implementation = 1, 

                             return_sequences = False, 

                             return_state = False, 

                             go_backwards = False, 

                             stateful = False, 

                             unroll = False)))



#model.add(GRU(units = 32, dropout = 0.2, recurrent_dropout = 0.2))

model.add(Dense(1, activation = 'sigmoid'))

model.summary()

plot_model(model, to_file = 'model_plot.png', show_shapes = True, show_layer_names = True)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# Training 



VALIDATION_SPLIT = 0.2

TEST_SPLIT = 0.2



indices = np.arange(review_pad.shape[0])

np.random.shuffle(indices)

review_pad = review_pad[indices]

sentiment = sentiment[indices]

num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])

num_test_samples = int(TEST_SPLIT * review_pad.shape[0])



train_size = review_pad.shape[0] - (num_validation_samples + num_test_samples)

X_train_pad = review_pad[0:train_size,:]

y_train = sentiment[0:train_size]

print(X_train_pad.shape)

X_val_pad = review_pad[train_size: (train_size + num_validation_samples), :]

y_val = sentiment[train_size: (train_size + num_validation_samples)]

print(X_val_pad.shape)

X_test_pad = review_pad[(train_size + num_validation_samples):review_pad.shape[0],:]

y_test = sentiment[(train_size + num_validation_samples):review_pad.shape[0]]

print(X_test_pad.shape)



print(indices.shape)
import tensorflow

config = tensorflow.ConfigProto(log_device_placement=True)

from keras.callbacks import EarlyStopping



print('Train...')

es = EarlyStopping(monitor = 'val_acc')

model.fit(X_train_pad, y_train, batch_size = 128, epochs = 75, validation_data = (X_val_pad, y_val), verbose = 2, callbacks = [es])
from sklearn.metrics import accuracy_score



ynew_pred = model.predict(X_test_pad)

ynew_pred[ynew_pred > 0.5] = 1

ynew_pred[ynew_pred < 0.5] = 0

acc = accuracy_score(y_test, ynew_pred, normalize = True)

print(acc)
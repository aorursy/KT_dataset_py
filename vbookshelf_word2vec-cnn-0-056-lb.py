import pandas as pd

import numpy as np



import math

from sklearn.model_selection import train_test_split



import nltk



from gensim.models import word2vec



from numpy import asarray

from numpy import zeros

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Embedding

from keras.optimizers import Adam

from keras.layers import BatchNormalization, Flatten, Conv1D, MaxPooling1D

from keras.layers import Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



# Don't Show Warning Messages

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



df_train.fillna(value='none',inplace=True)

df_test.fillna(value='none',inplace=True)



print(df_train.shape)

print(df_test.shape)
# combine the train and test sets for encoding and padding

train_len = len(df_train)

df_combined =  pd.concat(objs=[df_train, df_test], axis=0).reset_index(drop=True)



print(df_combined.shape)



# make a copy of df_combined

df_combined_copy = df_combined


# initialize the tokenizer for extracting sentences

tok = nltk.data.load('tokenizers/punkt/english.pickle')



output_list = []



def sentence_to_list(x):

    """

    1. Input: All text in the corpus - i.e. every comment

    2. Output: List of sentences where each sentence is a list of words e.g.

    output = [[hello,how,are,you],[i,am,fine,thanks]]

    3. The output python list contains all sentences from every train and test comment.

    

    """

    sentence_list= tok.tokenize(x)

    

    for sentence in sentence_list:

        # convert the sentence into a list of words

        word_list = sentence.split()

        # add the sentence to the list of sentences

        output_list.append(word_list)

        

    return output_list





# Run the function

# note that df_combined_copy['comment_text'] is not usable after this step.

# After running this line, a variable called output_list is created in memory...

# Okay, this is not the most pythonic way of doing things but apply() runs fast.

df_combined_copy['comment_text'].apply(sentence_to_list)



print(len(output_list))


# Set values for various parameters

num_features = 300    # Word vector dimensionality                      

min_word_count = 4    # Minimum word count                        

num_workers = 4       # Number of threads to run in parallel

context = 10          # Context window size                                                                                    

downsampling = 1e-3   # Downsample setting for frequent words



# Initialize and train the model



w2v_model = word2vec.Word2Vec(output_list, workers=num_workers, 

            size=num_features, min_count = min_word_count, 

            window = context, sample = downsampling)



w2v_model.init_sims(replace=True)



# save the model

# model_name = "300features_4minwords_10context"

# w2v_model.save(model_name)



print('Training completed.')
# save the word vectors



#word_vectors = w2v_model.wv

#word_vectors.save('word2vec_toxic_vectors.csv')



# load the saved word vectors

#word_vectors = KeyedVectors.load('word2vec_toxic_vectors.csv')

# get the shape of the word2vec embedding matrix

w2v_model.syn1neg.shape
# Tell me what words are most similar to the word 'man'?

w2v_model.most_similar("man")
# This is how to access the embedding vector for a given word

w2v_model.wv['hello']
# create the padded vectors



docs_combined = df_combined['comment_text'].astype(str)



# This tokenizer creates a python list of words

t = Tokenizer()

t.fit_on_texts(docs_combined)

vocab_size = len(t.word_index) + 1



# integer encode the documents

# assign each word a unique integer

encoded_docs = t.texts_to_sequences(docs_combined)



# pad documents to a max length of 500 words

max_length = 500 ###

padded_docs_combined = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# seperate the train and test sets



df_train_padded = padded_docs_combined[:train_len]

df_test_padded = padded_docs_combined[train_len:]



print(df_train_padded.shape)

print(df_test_padded.shape)
# create a embedding matrix for words that are in our combined train and test dataframes



embedding_matrix = zeros((vocab_size, 300))



for word, i in t.word_index.items():

    # check if the word is in the word2vec vocab

    if word in w2v_model.wv:

        embedding_vector = w2v_model.wv[word]

        

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector
# check the shape of the new embedding matrix

embedding_matrix.shape
X = df_train_padded

X_test = df_test_padded



# target columns

y_toxic = df_train['toxic']

y_severe_toxic = df_train['severe_toxic']

y_obscene = df_train['obscene']

y_threat = df_train['threat']

y_insult = df_train['insult']

y_identity_hate = df_train['identity_hate']
# target columns for each of the 6 models

target_cols = [y_toxic,y_severe_toxic,y_obscene,y_threat,y_insult,y_identity_hate]



preds = []



for col in target_cols:

    

    # set the value of y_train

    y = col

    

    X_train, X_eval, y_train ,y_eval = train_test_split(X, y,test_size=0.25,shuffle=True,

                                                    random_state=5,stratify=y)



    # define model

    model = Sequential()

    e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=500, trainable=False)

    model.add(e)

    model.add(Conv1D(128, 3, activation='relu'))

    model.add(MaxPooling1D(pool_size=3, strides=2))

    model.add(Dropout(0.2))

    model.add(Conv1D(64, 3, activation='relu'))

    model.add(MaxPooling1D(pool_size=3, strides=2))

    model.add(Dropout(0.2))

    model.add(Conv1D(64, 3, activation='relu'))

    model.add(MaxPooling1D(pool_size=3, strides=2))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))





    # compile the model

    Adam_new = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=Adam_new, loss='binary_crossentropy', metrics=['acc'])



    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

   

    save_best = ModelCheckpoint('toxic.hdf', save_best_only=True, monitor='val_loss', 

                               mode='min')



    history = model.fit(X_train, y_train, validation_data=(X_eval, y_eval),epochs=100, verbose=1,

                   callbacks=[early_stopping,save_best])





    model.load_weights(filepath = 'toxic.hdf')

    

    # make a prediction

    predictions = model.predict(X_test)



    y_preds = predictions[:,0]

    

    preds.append(y_preds)

# put the results into a dataframe



df_results = pd.DataFrame({'id':df_test.id,

                            'toxic':preds[0],

                           'severe_toxic':preds[1],

                           'obscene':preds[2],

                           'threat':preds[3],

                           'insult':preds[4],

                           'identity_hate':preds[5]}).set_index('id')



# Pandas automatically sorts the columns alphabetically by column name.

# Therefore we need to re-order the columns to match the sample submission file.

df_results = df_results[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]



# create a submission csv file

#df_results.to_csv('word2vec_with_cnn.csv', columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate']) 
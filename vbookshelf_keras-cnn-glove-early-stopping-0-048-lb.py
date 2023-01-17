import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split



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
# read in the data



#df_train = pd.read_csv('train.csv.zip')

#df_test = pd.read_csv('test.csv.zip')



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



print(df_train.shape)

print(df_test.shape)
# combine the train and test sets for encoding and padding



train_len = len(df_train)

df_combined =  pd.concat(objs=[df_train, df_test], axis=0).reset_index(drop=True)



print(df_combined.shape)
# define text data

docs_combined = df_combined['comment_text'].astype(str)



# initialize the tokenizer

t = Tokenizer()

t.fit_on_texts(docs_combined)

vocab_size = len(t.word_index) + 1



# integer encode the text data

encoded_docs = t.texts_to_sequences(docs_combined)



# pad the vectors to create uniform length

padded_docs_combined = pad_sequences(encoded_docs, maxlen=500, padding='post')
# seperate the train and test sets



df_train_padded = padded_docs_combined[:train_len]

df_test_padded = padded_docs_combined[train_len:]



print(df_train_padded.shape)

print(df_test_padded.shape)
# load the glove840B embedding into memory after downloading and unzippping



embeddings_index = dict()

f = open('glove.840B.300d.txt')



for line in f:

    # Note: use split(' ') instead of split() if you get an error.

	values = line.split(' ')

	word = values[0]

	coefs = np.asarray(values[1:], dtype='float32')

	embeddings_index[word] = coefs

f.close()



print('Loaded %s word vectors.' % len(embeddings_index))



# create a weight matrix

embedding_matrix = np.zeros((vocab_size, 300))

for word, i in t.word_index.items():

	embedding_vector = embeddings_index.get(word)

	if embedding_vector is not None:

		embedding_matrix[i] = embedding_vector
X = df_train_padded

X_test = df_test_padded



# target columns

y_toxic = df_train['toxic']

y_severe_toxic = df_train['severe_toxic']

y_obscene = df_train['obscene']

y_threat = df_train['threat']

y_insult = df_train['insult']

y_identity_hate = df_train['identity_hate']
# create a list of the target columns

target_cols = [y_toxic,y_severe_toxic,y_obscene,y_threat,y_insult,y_identity_hate]



preds = []



for col in target_cols:

    

    print('\n')

    

    # set the value of y

    y = col

    

    # create a stratified split

    X_train, X_eval, y_train ,y_eval = train_test_split(X, y,test_size=0.25,shuffle=True,

                                                    random_state=5,stratify=y)



    # cnn model

    model = Sequential()

    e = Embedding(vocab_size, 300, weights=[embedding_matrix], 

                  input_length=500, trainable=False)

    model.add(e)

    model.add(Conv1D(128, 3, activation='relu'))

    model.add(MaxPooling1D(3))

    model.add(Dropout(0.2))

    model.add(Conv1D(64, 3, activation='relu'))

    model.add(MaxPooling1D(3))

    model.add(Dropout(0.2))

    model.add(Conv1D(64, 3, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))





    # compile the model

    Adam_opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=Adam_opt, loss='binary_crossentropy', metrics=['acc'])



    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    save_best = ModelCheckpoint('toxic.hdf', save_best_only=True, 

                               monitor='val_loss', mode='min')



    history = model.fit(X_train, y_train, validation_data=(X_eval, y_eval),

                        epochs=100, verbose=1,callbacks=[early_stopping,save_best])



    

    # make a prediction on y (target column)

    model.load_weights(filepath = 'toxic.hdf')

    predictions = model.predict(X_test)

    y_preds = predictions[:,0]

    

    # append the prediction to a python list

    preds.append(y_preds)

df_results = pd.DataFrame({'id':df_test.id,

                            'toxic':preds[0],

                           'severe_toxic':preds[1],

                           'obscene':preds[2],

                           'threat':preds[3],

                           'insult':preds[4],

                           'identity_hate':preds[5]}).set_index('id')



# Pandas automatically sorts the columns alphabetically by column name.

# Therefore, we need to re-order the columns to match the sample submission file.

df_results = df_results[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]



# create a submission csv file

df_results.to_csv('kaggle_submission.csv', 

                  columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate']) 
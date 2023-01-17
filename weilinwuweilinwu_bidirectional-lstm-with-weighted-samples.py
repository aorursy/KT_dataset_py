import gc

import numpy as np

import pandas as pd

from typing import List, Tuple, Dict

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler
# a list of paths of embedding matrix files

# Two embedding matrixes will be combined together and used in the embedding layer of the RNN.

# They are CRAWL and GLOVE. Each of them is a collection of 300-dimension vector.

# Each vector represents a word.

# The coverage of words of CRAWL is different from that of GLOVE.

EMBEDDING_FILES = [

    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]



# <NUM_MODELS> represents the amount of times the same model should be trained

# Although each training is using the same RNN model, the predictions will be slightly different

# from each other due to different initialization (He Initialization).

NUM_MODELS = 2



# amount of epoch during training, this number is mainly limited by the GPU quota during committing.

EPOCHS = 4



# batch size

BATCH_SIZE = 256



# amount of LSTM units in each LSTM layer

LSTM_UNITS = 128



# amount of unit in the dense layer

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS



# maximum length of one comment (one sample)

MAX_LEN = 220



# column names related to identity in the training set

IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]



# a list of all the label names (Each sample/comment corresponds to multiple labels.)

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']



# column name of the comment column

TEXT_COLUMN = 'comment_text'



# target column

TARGET_COLUMN = 'target'



# chars to remove in the comment

# These chars are not covered by the embedding matrix. 

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
def get_coefs(word: str, *arr: str) -> (str, np.ndarray):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path: str) -> Dict[str, np.ndarray]:

    """Return a dict by analyzing the embedding matrix file under the path <path>."""

    

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(word_index: Dict[str, int], path: str, indexesOfWordsContainTrump: List[int]) -> np.ndarray:

    """Return an embedding matrix, which is ready to put into the RNN's embedding-matrix layer.

    

    <word_index>: Each word corresponds to a unique index. A word's vector can be found in the embedding

        matrix using the word's index.

    <path>: The path where the embedding matrix file is located at.

    <indexesOfWordsContainTrump>: A list of indexes of words that contain substring "Trump" or "trump".

    """

    

    # get a word-to-vector Dict by analyzing the embedding matrix file under the path <path>

    embedding_dict = load_embeddings(path)

    

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    

    # fill the <embedding_matrix> according to <embedding_dict>

    # If a tocken/word/string contains substring "Trump" or "trump", set the tocken/word/string's

    # vector to be the same as Trump's.

    # If a tocken/word cannot be found in <embedding_dict>, the tocken/word's vector is set to be zeros.

    # Otherwise, copy a tocken/word's vector from <embedding_dict> to <embedding_matrix>.

    for word, i in word_index.items():

        if(i in indexesOfWordsContainTrump):

            embedding_matrix[i] = embedding_dict['Trump']

        else:

            try:

                embedding_matrix[i] = embedding_dict[word]

            except KeyError:

                pass

            

    return embedding_matrix



def build_model(embedding_matrix: np.ndarray) -> Model:

    """Return a RNN model, which uses bidirectional LSTM."""

    

    # input layer

    words = Input(shape=(None,))

    

    # embedding matrix layer-this layer should be set to be not trainable.

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    

    # The dropout operation is used to prevent overfitting.

    x = SpatialDropout1D(0.2)(x)

    

    # two bidirectional LSTM layer

    # Since it is bidirectional, the output's size is twice the input's. 

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)



    # flatten the tensor by max pooling and average pooling

    # Since it is a concatenation of two pooling layer, the output's size is twice the input's.

    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    

    # two dense layers, skip conections trick is used here to prevent gradient's vanishing.

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    

    # two different output layers

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(len(AUX_COLUMNS), activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam')



    return model
# get the training set and test set offered in this competition

train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')



# seperate the targets and the features

x_train = train_df[TEXT_COLUMN].astype(str)

y_train = train_df[TARGET_COLUMN].values

y_aux_train = train_df[AUX_COLUMNS].values

x_test = test_df[TEXT_COLUMN].astype(str)



# change to continuous target values into discrete target values

# There are multiple targets. Each of them contains two different classes. They are True and False.

for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:

    train_df[column] = np.where(train_df[column] >= 0.5, True, False)



# One drawback of using Tokenizer is that it will change all characters to lower case.

# But the words in both CRAW and GLOVE are case sensitive.

# For example, "Trump" and "trump" are represented by different vectors in CRAWL or GLOVE.

# But Tokenizer will change "Trump" into "trump".

tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)



# A word-to-index Dict will be generated internally after analyzing the train set and the test set.

tokenizer.fit_on_texts(list(x_train) + list(x_test))



# Replace all the words/tokens in train/test set by the corresponding index according to the

# internal word-to-index Dict.

x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)



# make the length of all the sequencess the same

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
# assign different weights to different samples according to their labels

# This is because different groups have different effect on the evaluation metric.

# Another reason is that the evaluation metric is too complicated to be directly used during optimization.

# The following specific weight assignment is decided after many tries.

sample_weights = np.ones(len(x_train), dtype=np.float32)

sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)

sample_weights += train_df[TARGET_COLUMN] & (~train_df[IDENTITY_COLUMNS]).sum(axis=1)

sample_weights += (~train_df[TARGET_COLUMN]) & train_df[IDENTITY_COLUMNS].sum(axis=1) 

sample_weights += (~train_df[TARGET_COLUMN] & train_df['homosexual_gay_or_lesbian'] + 0) * 5

sample_weights += (~train_df[TARGET_COLUMN] & train_df['black'] + 0) * 5

sample_weights += (~train_df[TARGET_COLUMN] & train_df['white'] + 0) * 5

sample_weights += (~train_df[TARGET_COLUMN] & train_df['muslim'] + 0) * 1

sample_weights += (~train_df[TARGET_COLUMN] & train_df['jewish'] + 0) * 1

sample_weights /= sample_weights.mean()
indexesOfWordsContainTrump = []



# find out all the indexes of the words that contain substring "Trump" or "trump"

for word, index in tokenizer.word_index.items():

    if(('trump' in word) or ('Trump' in word)):

        indexesOfWordsContainTrump.append(index)



# The final embedding matrix is a concatenation of CRAWL embedding matrix and GLOVE embedding matrix.

# So each word is represented by a 600-d vector.

# In the final matrix, the words that contain substring "Trump" or "trump" are replaced by "Trump".

# This is found to be able to enhance the model performance by EDA(exploratory data analysis).

# The reason behind this is that strings like "Trump" and "trumpist" are related to toxicity,

# but they are covered neither in CRAWL nor GLOVE.

embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, filePath, indexesOfWordsContainTrump) for filePath in EMBEDDING_FILES], axis=-1)
# release memory space by deleting variables that are no longer useful

del train_df

del tokenizer

gc.collect()
# <checkpoint_predictions> is a list of predictions generated after each epoch.

checkpoint_predictions = []

# <weights> is a list of weights corresponding to <checkpoint_predictions>.

weights = []



for model_idx in range(NUM_MODELS):

    model = build_model(embedding_matrix)

    for global_epoch in range(EPOCHS):

        model.fit(

            x_train,

            [y_train, y_aux_train],

            batch_size=BATCH_SIZE,

            epochs=1,

            verbose=2,

            sample_weight=[sample_weights.values, np.ones_like(sample_weights)],

            callbacks=[

                LearningRateScheduler(lambda _: 1e-3 * (0.6 ** global_epoch))

            ]

        )

        

        # record predictions after each epoch

        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

        # Since the predictions tend to be more accurate after more epochs,

        # the weights is set to grow exponetially.

        weights.append(2 ** global_epoch)



# get the weighted average of the predictions. The average operation can help prevent overfitting.

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
# output the averaged predictions to a file for submission

submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    'prediction': predictions

})

submission.to_csv('submission.csv', index=False)
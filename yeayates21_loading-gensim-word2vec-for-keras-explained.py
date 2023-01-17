# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.preprocessing import text, sequence
# if we had more than 1 embedding file, we could list out the files

EMBEDDING_FILES = [

    '../input/jigsaw-custom-word2vec-100d-5iter/custom_word2vec_100d_5iter.txt'

]

# EMBEDDING_FILES = [

#     '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

#     '../input/glove840b300dtxt/glove.840B.300d.txt'

# ]



# if we have characters we want to remove before we tokenize, we can list them in a string

# CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
# going to use this just to stop the loop after a few rounds

rounds = 0



# core code

for path in EMBEDDING_FILES:

    with open(path) as f:

        for line in f:

            print(line)

            # end of core code

            # break after a few rounds

            rounds += 1

            if rounds==3:

                break
# going to use this just to stop the loop after a few rounds

rounds = 0



# core code

for path in EMBEDDING_FILES:

    with open(path) as f:

        for line in f:

            # main thing that's changing here is to strip each line and split it by a blank space

            print(line.strip().split(' '))

            # end of core code

            # break after a few rounds

            rounds += 1

            if rounds==3:

                break
# going to use this just to stop the loop after a few rounds

rounds = 0



# main thing that's changing here that we're going to pack the input into a tuple

def get_coefs(*mytup):

    return mytup



# core code

for path in EMBEDDING_FILES:

    with open(path) as f:

        for line in f:

            print(get_coefs(line.strip().split(' ')))

            # end of core code

            # break after a few rounds

            rounds += 1

            if rounds==3:

                break
# going to use this just to stop the loop after a few rounds

rounds = 0



# now we pack the data passed as a tuple

def get_coefs(*mytup):

    return mytup



# core code

for path in EMBEDDING_FILES:

    with open(path) as f:

        for line in f:

            # main thing that's changing here that we're going to unpack the input as a tuple

            print(get_coefs(*line.strip().split(' ')))

            # end of core code

            # break after a few rounds

            rounds += 1

            if rounds==3:

                break
# going to use this just to stop the loop after a few rounds

rounds = 0



# the main thing that's changing here is that we take the unpacked tuple and we take the first item as 'word' 

# and then we pack the rest as a tuple into 'coefs' and we return a tuple of word,coefs

def get_coefs(word, *coefs):

    return word, coefs



# core code

for path in EMBEDDING_FILES:

    with open(path) as f:

        for line in f:

            print(get_coefs(*line.strip().split(' ')))

            # end of core code

            # break after a few rounds

            rounds += 1

            if rounds==3:

                break
# going to use this just to stop the loop after a few rounds

rounds = 0



# the main thing that's changing here is we return coefs as a numpy array

def get_coefs(word, *coefs):

    return word, np.asarray(coefs, dtype='float32')



# core code

for path in EMBEDDING_FILES:

    with open(path) as f:

        for line in f:

            print(get_coefs(*line.strip().split(' ')))

            # end of core code

            # break after a few rounds

            rounds += 1

            if rounds==3:

                break
# going to use this just to stop the loop after a few rounds

rounds = 0



# the main thing that's changing here is we return coefs as a numpy array

def get_coefs(word, *coefs):

    return word, np.asarray(coefs, dtype='float32')



# core code

for path in EMBEDDING_FILES:

    with open(path) as f:

        embedding_index = dict(get_coefs(*line.strip().split(' ')) for line in f)

        # end of core code

        # break after a few rounds



# debug

for i in embedding_index:

    print(i)

    rounds += 1

    if rounds==3:

        break
# load data

x_train = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv')

x_train = x_train['comment_text'].astype(str)
# show some of the data

x_train[0:5]
# create tokenizer object

tokenizer = text.Tokenizer()

# if we wanted to remove characters we could run...

#tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(list(x_train))
my_word_index = tokenizer.word_index
# debugging

round = 0

for word, i in my_word_index.items():

    print(word,i)

    round += 1

    if round==3:

        break
len(my_word_index)
# we add a '+ 1' because this will create rows from 0-409327, i.e. up to but not including 409328, but 0-409327 in total is 409328 rows

# row 0 will be ignored for the most part and we'll fill 1-409327

embedding_matrix = np.zeros((len(my_word_index) + 1, 100))
embedding_matrix.shape
try:

    embedding_matrix[409328]

except:

    print("position does not exist")
# we can get the 0th row of the matrix using the following logic

embedding_matrix[0]
# we can get the coefficients for any word using the following logic

embedding_index['the']
# now we can put it all together, for loop through our tokenizer

# and for each (word,position) we can get the coefficents using the 'word'

# and we can set the coefficients to the correct position in the matrix using 'position'

for word, position in my_word_index.items():

    try:

        embedding_matrix[position] = embedding_index[word]

    except KeyError:

        pass
# debugging - showing that our matrix is loaded!

embedding_matrix[25]
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)





def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 100))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix
tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(x_train))
# if we have more than 1 EMBEDDING_FILE we can concatenate the results using np.concatenate()

embedding_matrix = np.concatenate([build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
# debugging - showing that our matrix is loaded!

embedding_matrix[25]
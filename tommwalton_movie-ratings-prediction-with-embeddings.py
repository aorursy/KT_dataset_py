# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# By running this at the beginning we can avoid readouts of deprecation warnings from tensorflow and other packages, 

# making the output more readable

import warnings

warnings.filterwarnings('ignore')
# numpy gives us the mathematical tools that we need

import numpy as np

# pandas lets us manipulate dataframes easily

import pandas as pd

# string helps us to remove punctuation from the overviews

import string 

# ast allows us to interpret strings literally - which helps us view a list of json files that had been stored as a string

# as a list instead.

import ast



# These nltk functions are used in preprocessing

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer



# This is a quick and easy function to separate out the test and train data using a randomiser to shuffle the source

from sklearn.model_selection import train_test_split



# We use these keras functions in our ML model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout

from tensorflow.keras.preprocessing import sequence



# matplotlib allows us to plot the progress of training on a graph

import matplotlib.pyplot as plt
movies = pd.read_csv("/kaggle/input/the-movies-dataset/movies_metadata.csv")

ratings = ratings = pd.read_csv("/kaggle/input/the-movies-dataset/ratings.csv")

credits = pd.read_csv("/kaggle/input/the-movies-dataset/credits.csv")



movies_subset = movies[["id", "overview"]]

# We need to be sure that the 'id' column only includes those where the id is a digit, as some appear to be in date format

# we also remove overviews of 30 or fewer characters to filter out blanks and generic "no overview found" listings 

movies_subset_2 = (movies_subset

                   .loc[movies_subset['id'].str.isdigit()]

                   .loc[movies_subset['overview'].str.len() > 30]

                  )

# We ensure that the id is parsed as an integer in all cases

movies_subset_2['id'] = movies_subset_2['id'].astype(int)



# We simply group all movies by id and take the mean of all ratings given to them

ratings_grouped = ratings.groupby('movieId').rating.mean()



overview_ratings = movies_subset_2.join(ratings_grouped, on="id")

credits_ratings = credits.join(ratings_grouped, on="id")



overview_ratings = overview_ratings.loc[overview_ratings['rating'] > 0]

credits_ratings = credits_ratings.loc[credits_ratings['rating'] > 0]
punctuation = list(string.punctuation)



stopWords = set(stopwords.words('english'))

ps = PorterStemmer()



def fix_overview(overview):

    cleansed_tokens = []

    

    for char in punctuation:

        if char in overview:

            overview = overview.replace(char, ' ')

        

    tokens = word_tokenize(overview)

    

    for word in tokens:

        if word not in stopWords:

            cleansed_tokens.append(ps.stem(word))

            

    return cleansed_tokens



overview_ratings['tokenized_overview'] = overview_ratings['overview'].map(fix_overview)
all_words = []



for x in overview_ratings['tokenized_overview'].values:

    for y in x:

        if y not in all_words:

            all_words.append(y)



words_dict = {}



for x in range(len(all_words)):

    words_dict[all_words[x]] = x + 1

    

def make_int_version(tokenized):

    int_version = []

    for i in tokenized:

        int_version.append(words_dict[i])

    

    return int_version

 

overview_ratings['int_tokens'] = overview_ratings['tokenized_overview'].map(make_int_version)
def tokenize_names(names):

    return [x['name'].lower() for x in ast.literal_eval(names)]



credits_ratings['cast_tokens'] = credits_ratings['cast'].map(tokenize_names)
all_names = []



for x in credits_ratings['cast_tokens'].values:

    for y in x:

        if y not in all_names:

            all_names.append(y)



names_dict = {}



for x in range(len(all_names)):

    names_dict[all_names[x]] = x + 1

    

def make_int_version(tokenized):

    int_version = []

    for i in tokenized:

        int_version.append(names_dict[i])

    

    return int_version

 

credits_ratings['int_tokens'] = credits_ratings['cast_tokens'].map(make_int_version)
def binarize(value):

    if value >= overview_ratings['rating'].mean():

        return 1

    return 0



overview_ratings['binary_rating'] = overview_ratings['rating'].map(binarize).astype(int)

credits_ratings['binary_rating'] = credits_ratings['rating'].map(binarize).astype(int)
x_1 = np.array(overview_ratings.loc[:, 'int_tokens'].values, dtype=object)

y_1 = np.array(overview_ratings.loc[:, 'binary_rating'].values, dtype=object)



x_2 = np.array(credits_ratings.loc[:, 'int_tokens'].values, dtype=object)

y_2 = np.array(credits_ratings.loc[:, 'binary_rating'].values, dtype=object)



x1_train, x1_val, y1_train, y1_val = train_test_split(x_1, y_1, train_size=0.7, random_state=24)

x2_train, x2_val, y2_train, y2_val = train_test_split(x_2, y_2, train_size=0.7, random_state=24)



x1_train = sequence.pad_sequences(x1_train, maxlen=134)

x1_val = sequence.pad_sequences(x1_val, maxlen=134)



x2_train = sequence.pad_sequences(x2_train, maxlen=313)

x2_val = sequence.pad_sequences(x2_val, maxlen=313)
model_1 = Sequential()

model_1.add(Embedding(134, 50))

model_1.add(LSTM(50))

model_1.add(Dense(1, activation='sigmoid'))

model_1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])



model_2 = Sequential()

model_2.add(Embedding(313, 50))

model_2.add(LSTM(50))

model_2.add(Dense(1, activation='sigmoid'))

model_2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
print("Training model 1...")

print()

history_1 = model_1.fit(x1_train, y1_train,

                   epochs = 10,

                   batch_size=128,

                   validation_split=0.2)



print()

print("Training model 2...")

print()



history_2 = model_2.fit(x2_train, y2_train,

                   epochs = 10,

                   batch_size=128,

                   validation_split=0.2)

import matplotlib.pyplot as plt



acc_1 = history_1.history['acc']

acc_2 = history_2.history['acc']



val_acc_1 = history_1.history['val_acc']

val_acc_2 = history_2.history['val_acc']



loss_1 = history_1.history['loss']

loss_2 = history_2.history['loss']



val_loss_1 = history_1.history['val_loss']

val_loss_2 = history_2.history['val_loss']



epochs = range(1, len(acc_1) + 1)
plt.plot(epochs, acc_1, 'bo', label='Training acc model 1')

plt.plot(epochs, val_acc_1, 'b', label='Validation acc model 1')

plt.plot(epochs, acc_2, 'ro', label='Training acc model 2')

plt.plot(epochs, val_acc_2, 'r', label='Validation acc model 2')



plt.title('Training and Validation accuracy between models 1 and 2')

plt.legend()



plt.figure()



plt.plot(epochs, loss_1, 'bo', label='Training loss model 1')

plt.plot(epochs, val_loss_1, 'b', label='Validation loss model 1')

plt.plot(epochs, loss_2, 'ro', label='Training loss model 2')

plt.plot(epochs, val_loss_2, 'r', label='Validation loss model 2')



plt.title('Training and Validation loss between models 1 and 2')

plt.legend()



plt.show()
import pandas as pd



df = pd.read_json("../input/Sarcasm_Headlines_Dataset.json", lines=True)

df.head()
import plotly as py

from plotly import graph_objs as go

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)



# Make pie chart to compare the numbers of sarcastic and not-sarcastic headlines

labels = ['Sarcastic', 'Not Sarcastic']

count_sarcastic = len(df[df['is_sarcastic']==1])

count_notsar = len(df[df['is_sarcastic']==0])

values = [count_sarcastic, count_notsar]

# values = [20,50]



trace = go.Pie(labels=labels,

               values=values,

               textfont=dict(size=19, color='#FFFFFF'),

               marker=dict(

                   colors=['#DB0415', '#2424FF'] 

               )

              )



layout = go.Layout(title = '<b>Sarcastic vs Not Sarcastic</b>')

data = [trace]

fig = go.Figure(data=data, layout=layout)



iplot(fig)
for i,headline in enumerate (df['headline'], 1):

    if i > 20:

        break

    else:

        print(i, headline)
import string

from string import digits, punctuation



hl_cleansed = []

for hl in df['headline']:

#     Remove punctuations

    clean = hl.translate(str.maketrans('', '', punctuation))

#     Remove digits/numbers

    clean = clean.translate(str.maketrans('', '', digits))

    hl_cleansed.append(clean)

    

# View comparison

print('Original texts :')

print(df['headline'][37])

print('\nAfter cleansed :')

print(hl_cleansed[37])

# Tokenization process

hl_tokens = []

for hl in hl_cleansed:

    hl_tokens.append(hl.split())



# View Comparison

index = 100

print('Before tokenization :')

print(hl_cleansed[index])

print('\nAfter tokenization :')

print(hl_tokens[index])
# Lemmatize with appropriate POS Tag

# Credit : www.machinelearningplus.com/nlp/lemmatization-examples-python/



import nltk

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer



def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}



    return tag_dict.get(tag, wordnet.NOUN)



# Init Lemmatizer

lemmatizer = WordNetLemmatizer()



hl_lemmatized = []

for tokens in hl_tokens:

    lemm = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]

    hl_lemmatized.append(lemm)

    

# Example comparison

word_1 = ['skyrim','dragons', 'are', 'having', 'parties']

word_2 = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_1]

print('Before lemmatization :\t',word_1)

print('After lemmatization :\t',word_2)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences

import numpy as np



# Vectorize and convert text into sequences

max_features = 2000

max_token = len(max(hl_lemmatized))

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(hl_lemmatized)

sequences = tokenizer.texts_to_sequences(hl_lemmatized)

X = pad_sequences(sequences, maxlen=max_token)
index = 10

print('Before :')

print(hl_lemmatized[index],'\n')

print('After sequences convertion :')

print(sequences[index],'\n')

print('After padding :')

print(X[index])



from sklearn.model_selection import train_test_split



Y = df['is_sarcastic'].values

Y = np.vstack(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, random_state = 42)
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D



embed_dim = 64



model = Sequential()

model.add(Embedding(max_features, embed_dim,input_length = max_token))

model.add(LSTM(96, dropout=0.2, recurrent_dropout=0.2, activation='relu'))

# model.add(Dense(128))

# model.add(Activation('relu'))

# model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
epoch = 10

batch_size = 128

model.fit(X_train, Y_train, epochs = epoch, batch_size=batch_size, verbose = 2)
loss, acc = model.evaluate(X_test, Y_test, verbose=2)

print("Overall scores")

print("Loss\t\t: ", round(loss, 3))

print("Accuracy\t: ", round(acc, 3))

pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

for x in range(len(X_test)):

    

    result = model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]

   

    if np.around(result) == np.around(Y_test[x]):

        if np.around(Y_test[x]) == 0:

            neg_correct += 1

        else:

            pos_correct += 1

       

    if np.around(Y_test[x]) == 0:

        neg_cnt += 1

    else:

        pos_cnt += 1
print("Sarcasm accuracy\t: ", round(pos_correct/pos_cnt*100, 3),"%")

print("Non-sarcasm accuracy\t: ", round(neg_correct/neg_cnt*100, 3),"%")
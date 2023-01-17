import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as plt

import tensorflow as tf

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation

from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D

from sklearn.model_selection import train_test_split

print(tf.__version__)
# Load data

#/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip

#/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip

#/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip

#/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip



train_df=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip').fillna(' ')

test_df=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip').fillna(' ')

train_df.sample(10)
x=train_df['comment_text'].values

x
# View few toxic comments

train_df.loc[train_df['toxic']==1]

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

comments = train_df['comment_text'].loc[train_df['toxic']==1].values

wordcloud = WordCloud(

    width = 640,

    height = 640,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(comments))

fig = plt.figure(

    figsize = (12, 8),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

comments = train_df['comment_text'].loc[train_df['toxic']==0].values

wordcloud = WordCloud(

    width = 640,

    height = 640,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(comments))

fig = plt.figure(

    figsize = (12, 8),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
y=train_df['toxic'].values

y
# Plot frequency of toxic comments



sns.countplot(x='toxic',data=train_df)

plt.title('Distribution of Toxic Comments')

train_df['toxic'].value_counts()
max_features=20000

max_text_length=400

x_tokenizer=text.Tokenizer(max_features)

x_tokenizer.fit_on_texts(list(x))

x_tokenized=x_tokenizer.texts_to_sequences(x)

x_train_val=sequence.pad_sequences(x_tokenized,maxlen=max_text_length)

#!wget http://nlp.stanford.edu/data/glove.6B.zip

#!unzip -q ./glove.6B.zip.1
embedding_dim=100

embedding_index=dict()

f=open('../input/glove6b100dtxt/glove.6B.100d.txt')

for line in f:

    values=line.split()

    word=values[0]

    coefs=np.asarray(values[1:],dtype='float32') 

    embedding_index[word]=coefs

    

f.close()

print(f'Found {len(embedding_index)} word vectors')
embedding_matrix=np.zeros((max_features,embedding_dim))

for word,index in x_tokenizer.word_index.items():

    if index>max_features-1:

        break

    else:

        embedding_vector=embedding_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[index]=embedding_vector
model=Sequential()

model.add(Embedding(max_features,

                   embedding_dim,

                   embeddings_initializer=tf.keras.initializers.Constant(

                   embedding_matrix),

                   trainable=False))

model.add(Dropout(0.2))

filters=250

kernel_size=3

hidden_dims=250

model.add(Conv1D(filters,

                kernel_size,

                padding='valid'))

model.add(MaxPooling1D())

model.add(Conv1D(filters,

                5,

                padding='valid',

                activation='relu'))

model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_dims,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])

x_train,x_val,y_train,y_val=train_test_split(x_train_val,y,test_size=0.2,random_state=1)
batch_size=32

epochs=3

model.fit(x_train,y_train,

         batch_size=batch_size,

         epochs=epochs,

         validation_data=(x_val,y_val))

test_df=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

test_df
x_test=test_df['comment_text'].values
x_test_tokenized=x_tokenizer.texts_to_sequences(x_test)

x_testing=sequence.pad_sequences(x_test_tokenized,maxlen=max_text_length)

y_testing=model.predict(x_testing,verbose=1,batch_size=32)

y_testing.shape
y_testing[0]
test_df['Toxic']=['Not Toxic' if x<0.5 else 'Toxic' for x in y_testing]

test_df[['comment_text','Toxic']].head(20)
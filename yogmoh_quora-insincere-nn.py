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
!pip install texthero
import pandas as pd

import numpy as np

import texthero as hero



import matplotlib.pyplot as plt

import re

import matplotlib as mpl



import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

mpl.rcParams['figure.dpi'] = 300
train = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')

test= pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')



display(train.sample(5))

display(train.info())

display(test.info())
combined_df = pd.concat([train.drop('target',axis=1),test])

combined_df.info()
hero.visualization.wordcloud(combined_df['question_text'], max_words=1000,background_color='BLACK')
combined_df['cleaned_text']=(combined_df['question_text'].pipe(hero.remove_angle_brackets)

                    .pipe(hero.remove_brackets)

                    .pipe(hero.remove_curly_brackets)

                    .pipe(hero.remove_diacritics)

                    .pipe(hero.remove_digits)

                    .pipe(hero.remove_html_tags)

                    .pipe(hero.remove_punctuation)

                    .pipe(hero.remove_round_brackets)

                    .pipe(hero.remove_square_brackets)

                    .pipe(hero.remove_stopwords)

                    .pipe(hero.remove_urls)

                    .pipe(hero.remove_whitespace)

                    .pipe(hero.lowercase))
lemm = WordNetLemmatizer()



def word_lemma(text):

    words = nltk.word_tokenize(text)

    lemma = [lemm.lemmatize(word) for word in words]

    joined_text = " ".join(lemma)

    return joined_text
combined_df['lemmatized_text'] = combined_df.cleaned_text.apply(lambda x: word_lemma(x))
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,f1_score,plot_confusion_matrix



from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Bidirectional

from tensorflow.keras.layers import Dropout
text = []

for i in range(len(combined_df)):

    review = nltk.word_tokenize(combined_df['lemmatized_text'].iloc[i])

    review = ' '.join(review)

    text.append(review)
voc_size = 5000

onehot_repr=[one_hot(words,voc_size)for words in text] 



sent_length=50

embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

display(embedded_docs)

display(embedded_docs.shape)



embedded_docs_train = embedded_docs[:1306122,:]

embedded_docs_test = embedded_docs[1306122:,:]
## Creating model

embedding_vector_features=150

model1=Sequential()

model1.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))

model1.add(Bidirectional(LSTM(100)))

model1.add(Dropout(0.3))

model1.add(Dense(1,activation='sigmoid'))

model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model1.summary())
X_final=np.array(embedded_docs_train)

y_final=np.array(train.target)



X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)



model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=100)
#Prediction on test data

y_pred1=model1.predict_classes(X_test)

print(classification_report(y_test,y_pred1))
y_pred2=model1.predict_classes(embedded_docs_test)
#Final predictions and submission

qid = test.qid



submissions = pd.DataFrame({'qid':qid,'target':y_pred2.reshape(-1)})

submissions.to_csv('./submission1.csv',index=False,header=True)
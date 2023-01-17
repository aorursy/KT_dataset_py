# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding,Bidirectional,BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

from keras.utils import  to_categorical

%matplotlib inline
from nltk.tokenize import word_tokenize 

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer() 

import re

import nltk
stopwords = nltk.corpus.stopwords.words('english')
def clean_text(text):

    

    text=re.sub(r"[^a-zA-Z]"," ",text)

    text=text.lower()

    #text=re.sub(r"[0-9]","",text)

    text=re.sub(r"i'm","i am",text)

    text=re.sub(r"he's","he is",text)

    text=re.sub(r"she's","she is",text)

    text=re.sub(r"that's","that is",text)

    text=re.sub(r"what's","what is",text)

    text=re.sub(r"where's","where is",text)

    text=re.sub(r"\'ll"," will",text)

    text=re.sub(r"\'ve"," have",text)

    text=re.sub(r"\'re"," are",text)

    text=re.sub(r"\'d"," would",text)

    text=re.sub(r"won't","will not",text)

    text=re.sub(r"can't","cannot",text)

    text=re.sub(r"[-()\"#/@;:<>{}+=~|.?,]","",text)

    word_tokens = word_tokenize(text)

    filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens if w not in stopwords]

    text=" ".join(filtered_sentence)

    return text
train=pd.read_csv('../input/incedodataclean/clean_text_apr26.csv')
train.head()
train['avg_score']=np.round((train.score_1+train.score_2+train.score_3)/3)
train.drop(columns=['score_1','score_2','score_3','score_4','score_5'],inplace=True)
train.head()
train2=train.drop(columns=['ID','Essayset','min_score','max_score','clarity','coherent'])
train2.head()
max_words = 2500

max_len = 50

tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(train2.EssayText)

sequences = tok.texts_to_sequences(train2.EssayText)

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

Y1=train2.avg_score
X_train,X_test,Y_train,Y_test = train_test_split(sequences_matrix,to_categorical(Y1),test_size=0.20)

def RNN():

    inputs = Input(name='inputs',shape=[max_len])

    layer = Embedding(max_words,50,input_length=max_len)(inputs)

    layer = Bidirectional(LSTM(64))(layer)

    #layer = Bidirectional(LSTM(128))(layer)

    layer = Dense(64,name='FC1')(layer)

    #layer=BatchNormalization()(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(64,name='FC2')(layer)

    #layer=BatchNormalization()(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(4,name='out_layer')(layer)

    layer = Activation('softmax')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model
model = RNN()

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=64,epochs=2,validation_data=(X_test,Y_test))
y_pred_train=model.predict(sequences_matrix)
tt1=pd.DataFrame(y_pred_train,columns=['out1','out2','out3','out4'])
tt1.hist()
train3=train.drop(columns=['ID','min_score','max_score','EssayText'])
train4=pd.concat([train3,tt1],axis=1)
train4.head()
import category_encoders as ce
ce1=ce.TargetEncoder(cols = ['Essayset','clarity','coherent'], min_samples_leaf = 20)
train4.loc[:,['Essayset','clarity','coherent']]=ce1.fit_transform(train4.loc[:,['Essayset','clarity','coherent']],train4.loc[:,['avg_score']])
train4.head()
X1=train4.drop(columns=['avg_score'])

Y1=train4.avg_score
from xgboost import XGBClassifier

xgb=XGBClassifier(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.01,

                 max_depth=4,

              

                 min_child_weight=0.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42) 
X1_train,X1_test,Y1_train,Y1_test=train_test_split(X1,Y1,test_size=0.2,random_state=123)
xgb.fit(X1_train,Y1_train)
yy_test=xgb.predict(X1_test)
from sklearn.metrics import  accuracy_score,classification_report
accuracy_score(Y1_test,yy_test)
print(classification_report(Y1_test,yy_test))
test=pd.read_csv('../input/incedonlpdata/incedo_nlpcadad7d/incedo_participant/test_dataset.csv')
test2=test.drop(columns=['ID','min_score','max_score'])
test2.head()
test2['EssayText']=test2['EssayText'].apply(clean_text)
sequences_test = tok.texts_to_sequences(test2.EssayText)

sequences_matrix_test = sequence.pad_sequences(sequences_test,maxlen=max_len)

test2_test=model.predict(sequences_matrix_test)
testdf=pd.DataFrame(test2_test,columns=['out1','out2','out3','out4'])
test3=pd.concat([test2,testdf],axis=1)
test3.drop(columns=['EssayText'],inplace=True)
test3.head()
test3.loc[:,['Essayset','clarity','coherent']]=ce1.transform(test3.loc[:,['Essayset','clarity','coherent']])
test3.head()
final_out=xgb.predict(test3)
out=pd.DataFrame()
out['id']=test.ID

out['essay_set']=test.Essayset

out['essay_score']=final_out

out.tail()
out.to_csv('incedo_nlp28aprv1.csv',index=None)
out.groupby('essay_set').essay_score.value_counts().plot(kind='bar',figsize=(50,10),fontsize=30)
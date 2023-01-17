
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import zipfile

with zipfile.ZipFile('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip') as z_test:
    z_test.extractall()
    
    
with zipfile.ZipFile('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip') as z_train:
    z_train.extractall()
    
with zipfile.ZipFile('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip') as z_sam:
    z_sam.extractall()
os.listdir()
train=pd.read_csv('train.csv',index_col=False)
test=pd.read_csv('test.csv',index_col=False)

train.sample(5)
test.sample(5)
import matplotlib.pyplot as plt
train.toxic.value_counts(normalize=True).plot.bar(title='toxic')
plt.show()
train.severe_toxic.value_counts(normalize=True).plot.bar(title='severe_toxic')
plt.show()
train.obscene.value_counts(normalize=True).plot.bar(title='obscene')
plt.show()
train.threat.value_counts(normalize=True).plot.bar(title='threat')
plt.show()
train.insult.value_counts(normalize=True).plot.bar(title='inslut')
plt.show()
train.identity_hate.value_counts(normalize=True).plot.bar(title='identity_hate')
plt.show()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,Dense,GlobalMaxPool1D,Dropout,Flatten,Bidirectional,LSTM
from keras.models import Sequential

train.sample(5)
num_words=(2000)
max_len=200
tokenizer=Tokenizer(2000)
tokenizer.fit_on_texts(train.comment_text)
train_sequences=tokenizer.texts_to_sequences(train.comment_text)
test_sequences=tokenizer.texts_to_sequences(test.comment_text)
padded_train=pad_sequences(train_sequences,maxlen=max_len)
padded_test=pad_sequences(test_sequences,maxlen=max_len)
y=train.iloc[:,2:].values
train_sequences[:1]
model=Sequential([Embedding(num_words,32,input_length=max_len),
                 Bidirectional(LSTM(32,return_sequences=True)),
                 GlobalMaxPool1D(),
                 Dense(32,activation='relu'),
                  Dense(6,activation='sigmoid')
                 ])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')
batch_size=20
epoch=2
history=model.fit(padded_train,y,batch_size,epochs=epoch,validation_split=.25,steps_per_epoch=300)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
test_ids=test.id
predicted=model.predict(padded_test)
predicted
cols=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
test['toxic']=predicted[:,:1]
test['severe_toxic']=predicted[:,1:2]
test['obscene']=predicted[:,2:3]
test['threat']=predicted[:,3:4]
test['insult']=predicted[:,4:5]
test['identity_hate']=predicted[:,5:6]

test
sample_sub=pd.read_csv('sample_submission.csv',index_col=False)
sample_sub.sample(5)
test.drop(['comment_text'],axis=1,inplace=True)
test
test.to_csv('toxic_comments_classification.csv',index=False)
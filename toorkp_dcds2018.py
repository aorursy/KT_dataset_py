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
# Worked with text part as of now. Still working on increasing the accuracy. will be updating soon taking other numerical parameters...
# valuable suggestions to improve the model/accuracy will be highly appreciated....

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout,Activation,Conv1D, MaxPooling1D
train = pd.read_csv('../input/train.csv')
#train.head()
train['text'] = train[['teacher_id',
                       'teacher_prefix',
                       'school_state',
                       #'project_submitted_datetime',
                       'project_grade_category',
                       'project_subject_categories',
                      'project_subject_subcategories',
                      'project_title',
                       'project_essay_1',
                       'project_essay_2',
                       'project_essay_3',
                       'project_essay_4',
                       'project_resource_summary']].apply(lambda x: ';'.join ([str(i) for i in x]), axis = 1)
#train['text'].head()
train.head()
train_drop = train.drop(columns = ['teacher_id',
                       'teacher_prefix',
                       'school_state',
                        'project_grade_category',
                        'project_subject_categories',
                        'project_subject_subcategories',
                        'project_title',
                        'project_essay_1',
                        'project_essay_2',
                        'project_essay_3',
                        'project_essay_4',
                        'project_resource_summary',
                        'project_submitted_datetime'])
train_drop.head(3)
train_drop.info()
#tokenizer = Tokenizer()#nb_words = MAX_NB_WORDS)
#vocab_size = 1000
tokenizer = Tokenizer()#(num_words = vocab_size)#nb_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(train_drop['text'])
X = tokenizer.texts_to_sequences(train_drop['text'])
X = pad_sequences(X)#, maxlen = MAX_SEQUENCE_LENGTH)
#train_drop.info()
y_train = train_drop['project_is_approved']
X_train = X #train_drop['text']#'teacher_number_of_previously_posted_projects','text']
ind = len(tokenizer.word_index)
ind
batch_size = 1024
epochs = 5 # 2
model = Sequential()
model.add(Embedding(input_dim=ind + 1, output_dim=30))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(30))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',metrics =['accuracy'])
model.fit(X_train,y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1)
#model.fit(X_train, y_train, batch_size=32, epochs=5)

model.summary()
test = pd.read_csv('../input/test.csv')
test['text'] = test[['teacher_id',
                       'teacher_prefix',
                       'school_state',
                       #'project_submitted_datetime',
                       'project_grade_category',
                       'project_subject_categories',
                      'project_subject_subcategories',
                      'project_title',
                       'project_essay_1',
                       'project_essay_2',
                       'project_essay_3',
                       'project_essay_4',
                       'project_resource_summary']].apply(lambda x: ';'.join ([str(i) for i in x]), axis = 1)
#train['text'].head()
#train.head()
test_drop = test.drop(columns = ['teacher_id',
                       'teacher_prefix',
                       'school_state',
                        'project_grade_category',
                        'project_subject_categories',
                        'project_subject_subcategories',
                        'project_title',
                        'project_essay_1',
                        'project_essay_2',
                        'project_essay_3',
                        'project_essay_4',
                        'project_resource_summary',
                        'project_submitted_datetime'])
test_drop.head(3)
X_test = tokenizer.texts_to_sequences(test_drop['text'])
X_test = pad_sequences(X_test)
pred = model.predict(X_test, batch_size = batch_size)
test['project_is_approved'] = pred[:,0]
test[['id','project_is_approved']].to_csv('submission.csv', index = False)
sub = pd.read_csv('submission.csv')
sub.head()






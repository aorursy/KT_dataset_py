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
import json
import re
import nltk
from bs4 import BeautifulSoup
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
nltk.download('stopwords')
from nltk.corpus import stopwords
print(os.listdir(".."))
print(os.listdir("../input"))
print(os.listdir("../input/word2vec-nlp-tutorial"))
raw_train_data = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", delimiter='\t')
raw_train_data.head()
raw_train_len = raw_train_data["review"].apply(len)
raw_train_len
plt.hist(raw_train_len, bins=300, color='g')
plt.yscale('log')
plt.title('Log-Scale Number of Reviews vs. Length of Reviews')
plt.xlabel('length of review')
plt.ylabel('number of review')
raw_train_len.describe()
fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6,3)
sns.countplot(raw_train_data['sentiment'])
print('number of + : {}'.format(raw_train_data['sentiment'].value_counts()[1]))
print('number of - : {}'.format(raw_train_data['sentiment'].value_counts()[0]))
train_word_cnt = raw_train_data['review'].apply(lambda x:len(x.split(' '))) # number of words in each review 
train_word_cnt
plt.figure(figsize=(8,5))
plt.hist(train_word_cnt, bins=50, color='g')
plt.yscale('log')
plt.title('Log-Scale Number of Reviews vs. Number of Reviews')
plt.xlabel('number of words')
plt.ylabel('number of reviews')
train_word_cnt.describe()
r_qmarks = np.mean(raw_train_data['review'].apply(lambda x: '?' in x))
r_fullstops = np.mean(raw_train_data['review'].apply(lambda x: '.' in x))
r_capitals = np.mean(raw_train_data['review'].apply(lambda x: max(y.isupper() for y in x)))
r_numbers = np.mean(raw_train_data['review'].apply(lambda x: max(y.isdigit() for y in x)))

print('물음표가 있는 리뷰 : {:.2f}%'.format(r_qmarks * 100))
print('마침표가 있는 리뷰 : {:.2f}%'.format(r_fullstops * 100))
print('대문자가 있는 리뷰 : {:.2f}%'.format(r_capitals * 100))
print('숫자가 있는 리뷰 : {:.2f}%'.format(r_numbers * 100))
review = raw_train_data['review'][0]
review_text = BeautifulSoup(review,"html5lib").get_text() # html 태그를 제거한다.
review_text = re.sub("[^a-zA-Z]"," ",review_text) # 알파벳을 제외하고 모두 공백으로 바꾼다.
print(review)
print(review_text)
stop_words = set(stopwords.words('english'))

review_text = review_text.lower()
words = review_text.split()
words = [w for w in words if not w in stop_words]
words[:10]
clean_review = ' '.join(words)
print(clean_review)
def preprocess(review,remove_stopwords = False):
    # html 제거
    review_text = BeautifulSoup(review,"html5lib").get_text()
    
    # 특수문자 제거
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    
    # 소문자로 통일 후 리스트화
    words = review_text.lower().split()
    
    if remove_stopwords:
        # 불용어 제거
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
 
    clean_review = ' '.join(words)
    
    return clean_review
clean_train_reviews = []
for review in raw_train_data['review']:
    clean_train_reviews.append(preprocess(review,remove_stopwords = True))
clean_train_reviews[0]
clean_train_df = pd.DataFrame({'id':raw_train_data['id'], 'review':clean_train_reviews, 'sentiment':raw_train_data['sentiment']})
clean_train_df
tokenizer = Tokenizer(oov_token='<UNK>')
#oov_token(out of vocab token)은 fitting된 tokenzier가 처음보는 단어를 어떻게 다룰지
#즉 사전에 없는 단어에 어떤 값을 취할건지 결정한다.
#본인은 <UNK>으로 설정하였으나 뭐로 하던 크게 상관없다.
tokenizer.fit_on_texts(clean_train_reviews)
text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)
print(clean_train_reviews[0])
print(text_sequences[0])
word_vocab = tokenizer.word_index
print(word_vocab)
#<UNK>의 인덱스가 1인것을 확인할 수 있다.
print(word_vocab["stuff"])
print("전체 단어 수: ",len(word_vocab))
#만약 oov_token을 추가하지 않을 경우 사전의 크기는 74066-1=74065가 된다.
data_configs = {}

data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab) + 1
MAX_SEQUENCE_LENGTH = 174
train_inputs = pad_sequences(text_sequences,maxlen=MAX_SEQUENCE_LENGTH, padding='post')

print('shape of train data: ', train_inputs.shape)
train_labels = np.array(raw_train_data['sentiment'])
print('shape of train labels: ',train_labels.shape)
DATA_IN_PATH = './data_in/'
TRAIN_INPUT_DATA = 'train_input.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
TRAIN_CLEAN_DATA = 'train_clean.csv'
DATA_CONFIGS = 'data_configs.json'

if not os.path.exists(DATA_IN_PATH):
    os.makedirs(DATA_IN_PATH)
np.save(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'wb'), train_inputs)
np.save(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'wb'), train_labels)

clean_train_df.to_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA, index = False)

json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w'), ensure_ascii=False)
test_data = pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv.zip", delimiter='\t')
test_data.head()
test_data
clean_test_reviews = []

for review in test_data['review']:
    clean_test_reviews.append(preprocess(review, remove_stopwords = True))
clean_test_df = pd.DataFrame({'review': clean_test_reviews, 'id': test_data['id']})
test_id = np.array(test_data['id'])

#여기서 테스트셋에 대해 tokenizer를 fitting 하지 않는다는 것을 유의하자.
text_sequences = tokenizer.texts_to_sequences(clean_test_reviews)
test_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
TEST_INPUT_DATA = 'test_input.npy'
TEST_CLEAN_DATA = 'test_clean.csv'
TEST_ID_DATA = 'test_id.npy'

np.save(open(DATA_IN_PATH + TEST_INPUT_DATA, 'wb'), test_inputs)
np.save(open(DATA_IN_PATH + TEST_ID_DATA, 'wb'), test_id)
clean_test_df.to_csv(DATA_IN_PATH + TEST_CLEAN_DATA, index = False)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, LSTM, Dropout, Bidirectional
vocab_size = len(word_vocab)+1
vocab_size
def small_model():
    model = Sequential()
    model.add(Embedding(vocab_size, 16))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model
model = small_model()
history = model.fit(train_inputs,
                    train_labels,
                    epochs=10,
                    batch_size=256,
                    validation_split = 0.3)
del model
del history
def big_model():
    model = Sequential()
    model.add(Embedding(vocab_size, 16))
    model.add(Bidirectional(LSTM(64, recurrent_dropout=0.1)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model
model = big_model()
history = model.fit(train_inputs,
                    train_labels,
                    epochs=10,
                    batch_size=256,
                    validation_split = 0.3)
model.save('my_model.h5')
%%time
test_label = model.predict_classes(test_inputs)
def submit(predictions):
    test_data['sentiment'] = predictions
    test_data.to_csv('submission.csv', index=False, columns=['id','sentiment'])

submit(test_label)

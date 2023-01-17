# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import gc



from sklearn.model_selection import KFold

from scipy.stats import spearmanr

from tqdm import tqdm, tqdm_notebook



from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



from keras.models import Model

from keras.models import Sequential

from keras import optimizers

from keras.layers import Embedding, Flatten, Dense,LeakyReLU, Input,concatenate, Dropout, GlobalMaxPooling1D,GlobalAveragePooling1D

from keras.layers import Activation,LSTM, Bidirectional,ReLU

from keras.callbacks import Callback

from keras.callbacks.callbacks import EarlyStopping





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pickle

embeddings = pickle.load( open( "/kaggle/input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl", "rb" ) )
dataPath = '/kaggle/input/google-quest-challenge/'

raw_train_data = pd.read_csv(dataPath+'train.csv')
print(raw_train_data.head(30))
sample_submission_df = pd.read_csv(dataPath+'sample_submission.csv')

print(sample_submission_df.columns)
target_labels = list(sample_submission_df.columns)

target_labels.remove('qa_id')

print(target_labels)
target_data = raw_train_data[target_labels]
print(['There are ' + str(target_data.shape[0]) + ' rows of data!'])
tqdm_notebook().pandas()

stop_words = set(stopwords.words('english'))

raw_train_data['question_title'] = raw_train_data['question_title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

raw_train_data['question_body'] = raw_train_data['question_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

raw_train_data['answer'] = raw_train_data['answer'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

raw_train_data['category'] = raw_train_data['category'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 

                       "'cause": "because", "could've": "could have", "couldn't": "could not", 

                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", 

                       "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", 

                       "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 

                       "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will",

                       "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", 

                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", 

                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 

                       "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", 

                       "ma'am": "madam", "mayn't": "may not", "might've": "might have",

                       "mightn't": "might not","mightn't've": "might not have", "must've": "must have", 

                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 

                       "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", 

                       "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", 

                       "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 

                       "she'll": "she will", "she'll've": "she will have", "she's": "she is", 

                       "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", 

                       "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", 

                       "that'd've": "that would have", "that's": "that is", "there'd": "there would", 

                       "there'd've": "there would have", "there's": "there is", "here's": "here is",

                       "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 

                       "they'll've": "they will have", "they're": "they are", "they've": "they have", 

                       "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", 

                       "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", 

                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 

                       "what're": "what are",  "what's": "what is", "what've": "what have", 

                       "when's": "when is", "when've": "when have", "where'd": "where did", 

                       "where's": "where is", "where've": "where have", "who'll": "who will", 

                       "who'll've": "who will have", "who's": "who is", "who've": "who have", 

                       "why's": "why is", "why've": "why have", "will've": "will have", 

                       "won't": "will not", "won't've": "will not have", "would've": "would have", 

                       "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 

                       "y'all'd": "you all would","y'all'd've": "you all would have",

                       "y'all're": "you all are","y'all've": "you all have","you'd": "you would", 

                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                       "you're": "you are", "you've": "you have" }

def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text

raw_train_data['question_title'] = raw_train_data['question_title'].apply(lambda x: clean_contractions(x, contraction_mapping))

raw_train_data['question_body'] = raw_train_data['question_body'].apply(lambda x: clean_contractions(x, contraction_mapping))

raw_train_data['answer'] = raw_train_data['answer'].apply(lambda x: clean_contractions(x, contraction_mapping))

raw_train_data['category'] = raw_train_data['category'].apply(lambda x: clean_contractions(x, contraction_mapping))
raw_train_data['question_title'] = raw_train_data.apply(lambda row: word_tokenize(row['question_title']), axis=1)

raw_train_data['question_body'] = raw_train_data.apply(lambda row: word_tokenize(row['question_body']), axis=1)

raw_train_data['answer'] = raw_train_data.apply(lambda row: word_tokenize(row['answer']), axis=1)

raw_train_data['category'] = raw_train_data.apply(lambda row: word_tokenize(row['category']), axis=1)
raw_test_data = pd.read_csv(dataPath+'test.csv')

raw_test_data['question_title'] = raw_test_data['question_title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

raw_test_data['question_body'] = raw_test_data['question_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

raw_test_data['answer'] = raw_test_data['answer'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

raw_test_data['category'] = raw_test_data['category'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

raw_test_data['question_title'] = raw_test_data['question_title'].apply(lambda x: clean_contractions(x, contraction_mapping))

raw_test_data['question_body'] = raw_test_data['question_body'].apply(lambda x: clean_contractions(x, contraction_mapping))

raw_test_data['answer'] = raw_test_data['answer'].apply(lambda x: clean_contractions(x, contraction_mapping))

raw_test_data['category'] = raw_test_data['category'].apply(lambda x: clean_contractions(x, contraction_mapping))

raw_test_data['question_title'] = raw_test_data.apply(lambda row: word_tokenize(row['question_title']), axis=1)

raw_test_data['question_body'] = raw_test_data.apply(lambda row: word_tokenize(row['question_body']), axis=1)

raw_test_data['answer'] = raw_test_data.apply(lambda row: word_tokenize(row['answer']), axis=1)

raw_test_data['category'] = raw_test_data.apply(lambda row: word_tokenize(row['category']), axis=1)
maxAllowedSequenceLength = 65

def text_to_array(textVal):

    emptyArr = np.zeros(300)    

    textVal = textVal[:maxAllowedSequenceLength]

    embed_text = [embeddings.get(text,emptyArr) for text in textVal]    

    embed_text+= [emptyArr] * (maxAllowedSequenceLength - len(embed_text))  

    return np.array(embed_text)
train = pd.DataFrame(columns = ['question_body','answer','question_title'])

test = pd.DataFrame(columns = ['question_body','answer','question_title'])

train['question_body'] = raw_train_data['question_body']

train['answer'] = raw_train_data['answer']

train['question_title'] = raw_train_data['question_title']

train['category'] = raw_train_data['category']

test['question_body'] = raw_test_data['question_body']

test['answer'] = raw_test_data['answer']

test['question_title'] = raw_test_data['question_title']

test['category'] = raw_test_data['category']

train['question_body'] = train.apply(lambda row: np.array(text_to_array(row['question_body'])), axis=1)

train['question_title'] = train.apply(lambda row: np.array(text_to_array(row['question_title'])), axis=1)

train['answer'] = train.apply(lambda row: np.array(text_to_array(row['answer'])), axis=1)

train['category'] = train.apply(lambda row: np.array(text_to_array(row['category'])), axis=1)

test['question_body'] = test.apply(lambda row: np.array(text_to_array(row['question_body'])), axis=1)

test['question_title'] = test.apply(lambda row: np.array(text_to_array(row['question_title'])), axis=1)

test['answer'] = test.apply(lambda row: np.array(text_to_array(row['answer'])), axis=1)

test['category'] = test.apply(lambda row: np.array(text_to_array(row['category'])), axis=1)
del embeddings,raw_test_data,raw_train_data

gc.collect()
class SpearmanRhoCallback(Callback):

    def __init__(self, X_train,y_train,X_val,y_val,model_lstm):

        self.x = X_train

        self.y = y_train.values

        self.x_val = X_val

        self.y_val = y_val.values

        self.model = model_lstm

        

    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return

        

    def on_epoch_begin(self, epoch,logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        rho_val = []

        val_pred = self.model.predict(self.x_val)

        shapeTarget = np.shape(self.y_val)

        for i_col in range(0,shapeTarget[1]):

            rho_val.append(spearmanr(self.y_val[:,i_col], val_pred[:,i_col]))

        rho_val = np.mean(rho_val)        

        return rho_val

    

    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return    
kf = KFold(n_splits=3, random_state=2019, shuffle=True)

y_test = []

for ind, (tr, val) in enumerate(kf.split(train)):

    X_train = train.loc[tr,:]

    y_train = target_data.loc[tr]

    X_val = train.loc[val,:]

    y_val = target_data.loc[val]

    inp1 = Input(shape=(maxAllowedSequenceLength,300))

    inp2 = Input(shape=(maxAllowedSequenceLength,300))

    inp3 = Input(shape=(maxAllowedSequenceLength,300))

    inp4 = Input(shape=(maxAllowedSequenceLength,300))

    inp = concatenate([inp1,inp2,inp3,inp4])

    x = Dense(2048,activation = 'relu')(inp)

    x = Dense(1024,activation = 'relu')(inp)

    hidden = concatenate([GlobalMaxPooling1D()(x),GlobalAveragePooling1D()(x)])

    hidden = Dense(256,activation = 'relu')(hidden)

    output = Dense(30,activation = 'sigmoid')(hidden)

    model_lstm = Model(inputs=[inp1,inp2,inp3,inp4], outputs=output)

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    model_lstm.compile(optimizer='adam',loss=['binary_crossentropy'])

    model_lstm.summary()

    model_lstm.fit([np.stack(X_train['question_title']),

                    np.stack(X_train['question_body']),

                    np.stack(X_train['answer']),

                    np.stack(X_train['category'])], 

                   y_train, epochs=100, batch_size=32,validation_data=(

                       [np.stack(X_val['question_title']),

                        np.stack(X_val['question_body']),

                        np.stack(X_val['answer']),

                        np.stack(X_val['category'])], 

                       y_val), 

                   verbose=True,callbacks = [es])

    y_test.append(model_lstm.predict([

        np.stack(test['question_title']),

        np.stack(test['question_body']),

        np.stack(test['answer']),

        np.stack(test['category'])

    ]))

    del model_lstm

    gc.collect()
y_out = np.mean(y_test,axis = 0)
sample_submission_df[target_labels] = np.squeeze(y_out)

sample_submission_df.head()

sample_submission_df.to_csv("submission.csv", index = False)
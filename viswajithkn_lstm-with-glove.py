# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/embeddings"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from keras.layers import Dense,Activation,LSTM, Bidirectional
from keras.models import Sequential

removeStopWords = 1
rawTrainData = pd.read_csv('../input/train.csv')
rawTestData = pd.read_csv('../input/test.csv')
print('The training data has the following: ',rawTrainData.columns)
print('The number of questions given in train data is: ',rawTrainData.shape[0])
def read_glove_vecs(glove_file):
    fileData = open(glove_file, 'r',encoding='utf-8')
    with fileData as f:
        words = set()
        word_to_vec_map = {}
        
        for line in tqdm(f):
            line = line.split(" ")
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float32)
            
    return words, word_to_vec_map
# read glove word vectors
words, word_to_vec_map = read_glove_vecs('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
print(rawTrainData.head(10))
from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()
from nltk.corpus import stopwords
def removeStopWords(word_list):
    return [word for word in word_list if word not in stopwords.words('english')]        
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(rawTrainData,test_size=0.02)
maxAllowedSequenceLength = 40
def text_to_array(textVal):
    emptyArr = np.zeros(300)    
    textVal = textVal[:maxAllowedSequenceLength]
    embed_text = [word_to_vec_map.get(text,emptyArr) for text in textVal]    
    embed_text+= [emptyArr] * (maxAllowedSequenceLength - len(embed_text))  
    return np.array(embed_text)
from nltk.tokenize import word_tokenize
val_df['tokenizedText'] = val_df.apply(lambda row: word_tokenize(row['question_text']), axis=1)
val_df['sents_length'] = val_df.apply(lambda row: len(row['tokenizedText']), axis=1)
val_df['stopWordsRemovedText'] = val_df.progress_apply(lambda row: removeStopWords(row['tokenizedText']), axis=1)
x_val = np.array([text_to_array(text) for text in tqdm(val_df['stopWordsRemovedText'][:])])
y_val = np.array(val_df['target'])
print(np.shape(x_val))
# I tried calling the text_to_array function directly on the train_df but ran out of memory after just a few tens of thousands
# of runs. So using generator and yield seems to be the way to go. let us see if that works.

batch_size = 256

def generateBatches(batch_size):
    numBatches = int(np.ceil(train_df.shape[0]/batch_size))
    while True:
        for i in range(numBatches):
            batchDF = train_df.iloc[i*batch_size : (i+1)*batch_size]
            batchDF['tokenizedText'] = batchDF.apply(lambda row: word_tokenize(row['question_text']), axis=1)
            batchDF['sents_length'] = batchDF.apply(lambda row: len(row['tokenizedText']), axis=1)            
            batchDF['stopWordsRemovedText'] = batchDF.progress_apply(lambda row: removeStopWords(row['tokenizedText']), axis=1)
            text_arr = np.array([text_to_array(text) for text in (batchDF['stopWordsRemovedText'])])
            targetVal = np.array(batchDF['target'])
            yield text_arr,targetVal                      
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                        input_shape=(40, 300)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', metrics = ['accuracy'],optimizer='adam')
dataGenerator = generateBatches(batch_size)
numBatches = np.ceil(train_df.shape[0]/batch_size)
model.fit_generator(dataGenerator,steps_per_epoch=500, epochs=6,validation_data = (x_val,y_val),verbose = True)
from sklearn.metrics import f1_score

pred_val_y = model.predict([x_val], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, f1_score(y_val, (pred_val_y>thresh).astype(int))))
rawTestData['tokenizedText'] = rawTestData.apply(lambda row: word_tokenize(row['question_text']), axis=1)
rawTestData['sents_length'] = rawTestData.apply(lambda row: len(row['tokenizedText']), axis=1)
rawTestData['stopWordsRemovedText'] = rawTestData.progress_apply(lambda row: removeStopWords(row['tokenizedText']), axis=1)  
def generateBatchesTest(batch_size):
    numBatches = int(np.ceil(rawTestData.shape[0]/batch_size))
    for i in range(numBatches):
        batchDF = rawTestData.iloc[i*batch_size : (i+1)*batch_size]
        text_arr = np.array([text_to_array(text) for text in (batchDF['stopWordsRemovedText'])])            
        yield text_arr      
pred_test_y = []
for x_test in tqdm(generateBatchesTest(batch_size)):
    pred_test_y.extend(model.predict(x_test,verbose=1))
y_pred = (np.array(pred_test_y)>0.33).astype(int)
submissionDF = pd.DataFrame({'qid':rawTestData['qid'],'prediction':y_pred.flatten()})
submissionDF.to_csv('submission.csv',index=False)
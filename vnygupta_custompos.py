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
train_data=pd.read_csv('../input/ml_assignment1/ML_Assignment1/pos tagger/train.txt',sep='\n',header=None) # data reading
train_data.head() #data sample
train_data.columns=['Text'] #column name assigning to text data
# reading tagged test data for evaluation purpose

Evaluate_data=pd.read_csv('../input/ml_assignment1/ML_Assignment1/pos tagger/test.tag',sep='\n',header=None) 
# Assigning column name and tagged data sample

Evaluate_data.columns=['Text']

Evaluate_data.head()
# function to seprate word and corresponding tag from line

def train_prep(line):

    sentance=[]

    line=str(line).replace('\t','')

    line=line.strip()

    line=line.split(" ")

    for word in line:

        sentance.append(tuple(word.split('/')))

    sent,tag=zip(*sentance)

    return sent,tag

    
# creating Input data and target data

Train_Sentance=[]

Train_Tag=[]

for line in train_data.Text:

    sent,tag=train_prep(line)

    Train_Sentance.append(np.array(sent))

    Train_Tag.append(np.array(tag))
from matplotlib import pyplot as plt
plt.plot([len(p)  for p in Train_Sentance]) # ploting number of words in each line
plt.hist([len(p)  for p in Train_Sentance]) # plotting distribution of  no of words in Sentences
# calculating mean and median of words in sentences and standard deviation ,so that we can minimize 

# the number of post padding in each line to equalize shape of lines

np.mean([len(p) for p in Train_Sentance]),np.std([len(p) for p in Train_Sentance]),np.median([len(p) for p in Train_Sentance])
Train_Sentance=[]

Train_Tag=[]

for line in train_data.Text:

    sent,tag=train_prep(line)

    if len(sent)<=40:

        Train_Sentance.append(np.array(sent))

        Train_Tag.append(np.array(tag))
# preparing evaluation data same as training data, only those sentences which has max 40 words

Eval_Sentance=[]

Eval_Tag=[]

i=0

index=[] # preserving index of lines which is taken evaluation for future reference

for line in Evaluate_data.Text:

    sent,tag=train_prep(line)

    if len(sent)<=40:

        index.append(i)

        Eval_Sentance.append(np.array(sent))

        Eval_Tag.append(np.array(tag))

    i+=1


words, tags = set([]), set([])

for s in Train_Sentance:

    for w in s:

        words.add(w.lower())

for ts in Train_Tag:

    for t in ts:

        tags.add(t)

        

word2index = {w: i + 2 for i, w in enumerate(list(words))}

word2index['-PAD-'] = 0  # The special value used for padding

word2index['-OOV-'] = 1  # The special value used for OOVs( Out of vacabulary)



tag2index = {t: i + 1 for i, t in enumerate(list(tags))}

tag2index['-PAD-'] = 0  # The special value used to padding
train_sentences_X,train_tags_y,eval_sentences_X,eval_tags_y = [],[],[],[]

# converting words to number in sentences for training

for s in Train_Sentance:

    s_int = []

    for w in s:

        try:

            s_int.append(word2index[w.lower()])

        except KeyError:

            s_int.append(word2index['-OOV-'])

 

    train_sentences_X.append(s_int)

# converting words to number in sentences for evaluation

for s in Eval_Sentance:

    s_int = []

    for w in s:

        try:

            s_int.append(word2index[w.lower()])

        except KeyError:

            s_int.append(word2index['-OOV-'])

 

    eval_sentences_X.append(s_int)

# converting tags to number in sentences for training   

for s in Train_Tag:

    train_tags_y.append([tag2index[t] for t in s])

# converting tags to number in sentences for evaluation    

for s in Eval_Tag:

    eval_tags_y.append([tag2index[t] for t in s])

    

print(train_sentences_X[0])

print(eval_sentences_X[0])

print(train_tags_y[0])

print(eval_tags_y[0])
MAX_LENGTH =40

from keras.preprocessing.sequence import pad_sequences

# Preprocessing the data to equalize shape of each input by padding 

train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')

eval_sentences_X = pad_sequences(eval_sentences_X, maxlen=MAX_LENGTH, padding='post')

train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')

eval_tags_y = pad_sequences(eval_tags_y, maxlen=MAX_LENGTH, padding='post')

 

print(train_sentences_X[0])

print(eval_sentences_X[0])

print(train_tags_y[0])

print(eval_tags_y[0])
# Function to convert target variable into categorical variable

def to_categorical(sequences, categories):

    tag_sequences = []

    for s in sequences:

        tags = []

        for item in s:

            tags.append(np.zeros(categories))

            tags[-1][item] = 1.0

        tag_sequences.append(tags)

    return np.array(tag_sequences)
from keras import backend as K

# function to calculate actual by excluding contribution of padding

def actual_class_accuracy(to_ignore=0):

    def actual_accuracy(y_true, y_pred):

        y_true_class = K.argmax(y_true, axis=-1)

        y_pred_class = K.argmax(y_pred, axis=-1)

 

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')

        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask

        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)

        return accuracy

    return actual_accuracy



# Function to calcualte reacll

def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall

# Function to calcualte prescision

def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

# Function to calcualte f1-score

def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# Model Building

from keras.models import Sequential

from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation

from keras.optimizers import Adam

 

model = Sequential()

model.add(InputLayer(input_shape=(MAX_LENGTH, )))

model.add(Embedding(len(word2index), 128))

model.add(Bidirectional(LSTM(256, return_sequences=True)))

model.add(TimeDistributed(Dense(len(tag2index))))

model.add(Activation('softmax'))

 

model.compile(loss='categorical_crossentropy',

              optimizer=Adam(0.001),

               metrics=['acc',actual_class_accuracy(0),f1_m,precision_m, recall_m])

 

model.summary()
from sklearn.model_selection import train_test_split # seperating training and validation data

X_train,X_test,Y_train,Y_test=train_test_split(train_sentences_X,to_categorical(train_tags_y, len(tag2index)),test_size=0.2)
# Evaltuation data

X_eval=eval_sentences_X

Y_eval=to_categorical(eval_tags_y, len(tag2index))
# Model Training

model.fit(X_train,Y_train, batch_size=64, epochs=12, validation_data=(X_test,Y_test))
# plot of training loss and validation loss

plt.plot(model.history.history['loss'],label="loss")

plt.plot(model.history.history['val_loss'],label="val_loss")

plt.legend()
# plot of training accuracy and validation accuracy

plt.plot(model.history.history['acc'],label="acc")

plt.plot(model.history.history['val_acc'],label="val_acc")

plt.legend()
#plot of training actual accuracy and validation actual acccuracy

plt.plot(model.history.history['actual_accuracy'],label="Actual Accuracy")

plt.plot(model.history.history['val_actual_accuracy'],label="val_Actual_accuracy")

plt.legend()
# plot of f1-score of training and valiadtion

plt.plot(model.history.history['f1_m'],label="f1_score")

plt.plot(model.history.history['val_f1_m'],label="val_f1_score")

plt.legend()
#plot of precision of training and valiadtion

plt.plot(model.history.history['precision_m'],label="precision_score")

plt.plot(model.history.history['val_precision_m'],label="val_precision_score")

plt.legend()
#plot of recall value of training and valiadtion 

plt.plot(model.history.history['recall_m'],label="recall_score")

plt.plot(model.history.history['val_recall_m'],label="val_recall_score")

plt.legend()
loss,accuracy,actutal_acc,f1_score,precision,recall=model.evaluate(X_eval,Y_eval)
print("Loss:{:.4f}".format(loss))

print("Accuracy:{:.4f}".format(accuracy))

print("Acutal_Accuracy:{:.4f}".format(actutal_acc))

print("F1-Score:{:.4f}".format(f1_score))

print("Precision:{:.4f}".format(precision))

print("Recall:{:.4f}".format(recall))
test_data=pd.read_csv('../input/ml_assignment1/ML_Assignment1/pos tagger/test.txt',sep='\n',header=None) # reading unlabelled text data
test_data.columns=['Text'] # assigning column name
# Function to prepare untagged text test data for cross-valiadtion

def test_prep(line):

    sentance=[]

    line=str(line).replace('\t','')

    line=line.strip()

    line=line.split(" ")

    return line

    
# Test data preparation for cross validation

Sentance_Test=[]

for i in index:

    sent=test_prep(test_data.Text[i])

    Sentance_Test.append(np.array(sent))

        
# converting words to nummber in each line for test

test_sentences_X= []

for s in Sentance_Test:

    s_int = []

    for w in s:

        try:

            s_int.append(word2index[w.lower()])

        except KeyError:

            s_int.append(word2index['-OOV-'])

    test_sentences_X.append(s_int)
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post') # padding test sentences
test_pred=model.predict(test_sentences_X) # predicting test tags
# Function to convert numbers to tag

def logits_to_tokens(sequences, index):

    token_sequences = []

    for categorical_sequence in sequences:

        token_sequence = []

        for categorical in categorical_sequence:

            token_sequence.append(index[np.argmax(categorical)])

 

        token_sequences.append(token_sequence)

 

    return token_sequences
# combining words and  associated predicted tags

Sentences_list=[]

p=0

for i in index:

    Sent=[]

    b=test_data.Text[i].split(" ")

    pos=logits_to_tokens(test_pred, {i: t for t, i in tag2index.items()})[p][:len(b)]

    for j in range(len(b)):

        Sent.append("/".join([b[j],pos[j]]))

    Sentences_list.append(" ".join(Sent))

    p+=1

    
for i in range(len(index)):

    print("Sentenc{:1d}   model's predicted output:\n".format(i))

    print(Sentences_list[i])

    print("\nSentenc{:1d} Actual data:\n".format(i))

    print(Evaluate_data.iloc[index[i],:].Text)

    print("\n")

    print("-"*120)

    print("\n")
test_data.to_csv("dummy.csv")
def no_jump(a):

    index=0

    b=len(a)

    jump=0

    while(index<b):

        index+=a[index]

        jump+=1

        if jump>b: # number  of jump cannot be grater the the length of arry

            return -1

    return jump
no_jump([2,3,-1,1,3])
no_jump([1,1,-1,1])
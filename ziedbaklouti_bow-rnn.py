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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS,CountVectorizer
train_set=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

train_set = train_set.fillna('-')
vect1 = CountVectorizer(token_pattern=r'[a-z|A-Z|**]{3,10}',ngram_range=(1, 1),max_features=23000, stop_words=ENGLISH_STOP_WORDS).fit(train_set.text)
X1_txt = vect1.transform(train_set.text)
X1=pd.DataFrame(X1_txt.toarray(), columns=vect1.get_feature_names())
y=train_set.sentiment
log_reg1 = LogisticRegression(C=1500).fit(X1, y)
y1_predicted = log_reg1.predict(X1)
test_set=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
X2_txt = vect1.transform(test_set.text)
X2=pd.DataFrame(X2_txt.toarray(), columns=vect1.get_feature_names())
y2_predicted = log_reg1.predict(X2)
from sklearn.metrics import accuracy_score
print('Accuracy score test set BOW: ', accuracy_score(train_set.sentiment, y1_predicted))
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
docs=[nlp(train_set.text[i]) for i in range(len(train_set.text))]
tokens = [[token.text.lower() for token in doc] for doc in docs]
tokens
a=train_set.sentiment.values == 'negative' 
b=train_set.sentiment.values=='positive'
A=[]

for x,y in zip(a,b):

    A.append(x or y)
tokens_pos_neg=[]



for i,x in enumerate(A):

    if x:

        tokens_pos_neg.append(tokens[i])

        
tokens_pos_neg
selected_text=list(train_set.selected_text.values)
selected_text=list(train_set.selected_text.values)
docs=[nlp(selected_text[i]) for i in range(len(selected_text))]
tokens = [[token.text.lower() for token in doc] for doc in docs]
selected_text_pos_neg=[]

for i,x in enumerate(A):

    if x:

        selected_text_pos_neg.append(tokens[i])
c=zip(tokens_pos_neg,selected_text_pos_neg)
k=[*c]
dd={}

for i,P in enumerate(k):

    d={}

    if len(P[0])>0 and len(P[1])>0:

        for rank,value in enumerate(P[0]):

            if rank<=len(P[0])-len(P[1]):

                L=[]

                for j in range(len(P[1])):

                    L.append(P[0][rank+j])

                d[rank]=L

    

    dd[i]=d
dd
detects=[]

for rank1,value1 in dd.items():

    detect=[]

    for rank,value in value1.items():

        if value==selected_text_pos_neg[rank1]:

            detect.append(rank)

            detect.append(value)

    detects.append(detect)
for rank1,value1 in enumerate(detects):

    if len(value1)>0  and value1[1]!=selected_text_pos_neg[rank1]:

        value1[1]=[]
tagget_extracted_tokens=[]

for rank1,value1 in enumerate(detects):

    if len(value1)>0 and len(tokens_pos_neg)>0:

        val=np.zeros(len(tokens_pos_neg[rank1]))

        #print(value1)

        #print(value1[0])

        #print(value1[1])

        if value1[0]!=0:

            val[value1[0]-1:(value1[0]-1)+len(value1[1])+1]=list(range(len(value1[1])+1))

        else:

            val[0:(value1[0]-1)+len(value1[1])+1]=range(len(value1[1]))+np.ones(len(value1[1]),dtype=int)

        tagget_extracted_tokens.append(list(val))

    else:

        tagget_extracted_tokens.append([])
tagget_extracted_tokens
for k,w in enumerate(tagget_extracted_tokens):

    w.insert(0,0)

    w.append(0)
for k,w in enumerate(tokens_pos_neg):

    w.insert(0,'#')

    w.append('#')
word2idx = {}

word_idx = 0
for k,w in enumerate(tokens_pos_neg):

    for k1,w1 in enumerate(w):

        if w1 not in word2idx:

            word2idx[w1]=word_idx

            word_idx += 1
word2idx
len_word2idx=len(word2idx)
len_word2idx
pos2idx = {}

pos_idx = 0

for k,w in enumerate(tagget_extracted_tokens):

    for k1,w1 in enumerate(w):

        if w1 not in pos2idx:

            pos2idx[w1]=pos_idx

            pos_idx += 1

pos2idx
Xtrain=[]

for k,w in enumerate(tokens_pos_neg):

    l=[]

    for k1,w1 in enumerate(w):

        #print(w1)

        #print(w1)

        l.append(word2idx[w1])

    Xtrain.append(l)
Xtrain
Ptrain=[]

for k,w in enumerate(tagget_extracted_tokens):

    l=[]

    for k1,w1 in enumerate(w):

        #print(w1)

        #print(w1)

        l.append(pos2idx[w1])

    Ptrain.append(l)
Ptrain


X=[]

P=[]





selected_text_pos_neg_modified=[]

for x,z,s in zip(Xtrain,Ptrain,selected_text_pos_neg):

    if len(x)==len(z):

        

        X.append(x)

        P.append(z)

        selected_text_pos_neg_modified.append(s)
len(X)
lenght_list=[]

for l in X:

    lenght_list.append(len(l))

max_len=np.max(lenght_list)
max_len
input_data=np.zeros((2000,max_len+1,len(word2idx)),dtype='float32')

target_data=np.zeros((2000,max_len+1,len(pos2idx)),dtype='float32')

for k,w in enumerate(X[0:2000]):

    for k1,w1 in enumerate(w):

        input_data[k,k1,X[k][k1]]=1

for k,w in enumerate(X[0:2000]):

    for k1,w1 in enumerate(w):

        target_data[k,k1,P[k][k1]]=1

from keras.layers import SimpleRNN,Dense,Activation,TimeDistributed

from keras.models import Sequential

model=Sequential()

model.add(SimpleRNN(50,input_shape=(max_len+1,len(word2idx)),return_sequences=True))

model.add(TimeDistributed(Dense(len(pos2idx),activation='softmax')))

model.compile(loss="categorical_crossentropy",optimizer="adam")

model.fit(input_data,target_data,batch_size=100,epochs=15)

for k2 in range(1,7):

    input_data=np.zeros((2000,max_len+1,len(word2idx)),dtype='float32')

    target_data=np.zeros((2000,max_len+1,len(pos2idx)),dtype='float32')

    for k,w in enumerate(X[k2*2000:k2*2000+2000]):

        for k1,w1 in enumerate(w):

            input_data[k,k1,X[k+k2*2000][k1]]=1

    for k,w in enumerate(X[k2*2000:k2*2000+2000]):

        for k1,w1 in enumerate(w):

            target_data[k,k1,P[k+k2*2000][k1]]=1

    model.fit(input_data,target_data,batch_size=100,epochs=15)

input_data=np.zeros((586,max_len+1,len(word2idx)),dtype='float32')

target_data=np.zeros((586,max_len+1,len(pos2idx)),dtype='float32')

k2=7

for k,w in enumerate(X[k2*2000:k2*2000+586]):

        for k1,w1 in enumerate(w):

            input_data[k,k1,X[k+k2*2000][k1]]=1

for k,w in enumerate(X[k2*2000:k2*2000+586]):

        for k1,w1 in enumerate(w):

            target_data[k,k1,P[k+k2*2000][k1]]=1

model.fit(input_data,target_data,batch_size=100,epochs=15)
len(tokens_pos_neg)
tokens_pos_neg_3=[]

tokens_pos_neg_2=[]

tokens_pos_neg_1=[]

tokens_pos_neg_33=[]

tokens_pos_neg_22=[]

tokens_pos_neg_11=[]

for k,w in enumerate(tokens_pos_neg):

    for k11,w11 in enumerate(w):

        w1=w11[:3]

        tokens_pos_neg_3.append(w1)

        w2=w11[:2]

        tokens_pos_neg_2.append(w2)

        w3=w11[:1]

        tokens_pos_neg_1.append(w3)

    tokens_pos_neg_33.append(tokens_pos_neg_3)

    tokens_pos_neg_22.append(tokens_pos_neg_2)

    tokens_pos_neg_11.append(tokens_pos_neg_1)

    tokens_pos_neg_3=[]

    tokens_pos_neg_2=[]

    tokens_pos_neg_1=[]
len(tokens_pos_neg_33)
word2idx_3 = {}

word_idx_3 = 0

for k,w in enumerate(tokens_pos_neg_33):

    for k1,w1 in enumerate(w):

        if w1 not in word2idx_3:

            word2idx_3[w1]=word_idx_3

            word_idx_3 += 1

print(word2idx_3)
len(word2idx_3)
len(Ptrain)
Xtrain_3=[]

for k,w in enumerate(tokens_pos_neg_33):

    l=[]

    for k1,w1 in enumerate(w):

        #print(w1)

        #print(w1)

        l.append(word2idx_3[w1])

    Xtrain_3.append(l)
len(Xtrain_3)
X3=[]

P3=[]

for x,z in zip(Xtrain_3,Ptrain):

    if len(x)==len(z):

        X3.append(x)

        P3.append(z)

        #selected_text_pos_neg_modified.append(s)
len(X3)
len(P3)
input_data=np.zeros((2000,max_len+1,len(word2idx_3)),dtype='float32')

target_data=np.zeros((2000,max_len+1,len(pos2idx)),dtype='float32')

for k,w in enumerate(X3[0:2000]):

    for k1,w1 in enumerate(w):

        input_data[k,k1,X3[k][k1]]=1

for k,w in enumerate(X3[0:2000]):

    for k1,w1 in enumerate(w):

        target_data[k,k1,P3[k][k1]]=1

from keras.layers import SimpleRNN,Dense,Activation,TimeDistributed

from keras.models import Sequential

model3=Sequential()

model3.add(SimpleRNN(50,input_shape=(max_len+1,len(word2idx_3)),return_sequences=True))

model3.add(TimeDistributed(Dense(len(pos2idx),activation='softmax')))

model3.compile(loss="categorical_crossentropy",optimizer="adam")

model3.fit(input_data,target_data,batch_size=100,epochs=15)

for k2 in range(1,7):

    input_data=np.zeros((2000,max_len+1,len(word2idx_3)),dtype='float32')

    target_data=np.zeros((2000,max_len+1,len(pos2idx)),dtype='float32')

    for k,w in enumerate(X3[k2*2000:k2*2000+2000]):

        for k1,w1 in enumerate(w):

            input_data[k,k1,X3[k+k2*2000][k1]]=1

    for k,w in enumerate(X3[k2*2000:k2*2000+2000]):

        for k1,w1 in enumerate(w):

            target_data[k,k1,P3[k+k2*2000][k1]]=1

    model3.fit(input_data,target_data,batch_size=100,epochs=15)

input_data=np.zeros((586,max_len+1,len(word2idx_3)),dtype='float32')

target_data=np.zeros((586,max_len+1,len(pos2idx)),dtype='float32')

k2=7

for k,w in enumerate(X3[k2*2000:k2*2000+586]):

        for k1,w1 in enumerate(w):

            input_data[k,k1,X3[k+k2*2000][k1]]=1

for k,w in enumerate(X3[k2*2000:k2*2000+586]):

        for k1,w1 in enumerate(w):

            target_data[k,k1,P3[k+k2*2000][k1]]=1

model3.fit(input_data,target_data,batch_size=100,epochs=15)
word2idx_2 = {}

word_idx_2 = 0

for k,w in enumerate(tokens_pos_neg_22):

    for k1,w1 in enumerate(w):

        if w1 not in word2idx_2:

            word2idx_2[w1]=word_idx_2

            word_idx_2 += 1

print(word2idx_2)
len(word2idx_2)
Xtrain_2=[]

for k,w in enumerate(tokens_pos_neg_22):

    l=[]

    for k1,w1 in enumerate(w):

        #print(w1)

        #print(w1)

        l.append(word2idx_2[w1])

    Xtrain_2.append(l)
X2=[]

P2=[]

for x,z in zip(Xtrain_2,Ptrain):

    if len(x)==len(z):

        X2.append(x)

        P2.append(z)

        #selected_text_pos_neg_modified.append(s)
input_data=np.zeros((2000,max_len+1,len(word2idx_2)),dtype='float32')

target_data=np.zeros((2000,max_len+1,len(pos2idx)),dtype='float32')

for k,w in enumerate(X2[0:2000]):

    for k1,w1 in enumerate(w):

        input_data[k,k1,X2[k][k1]]=1

for k,w in enumerate(X2[0:2000]):

    for k1,w1 in enumerate(w):

        target_data[k,k1,P2[k][k1]]=1

from keras.layers import SimpleRNN,Dense,Activation,TimeDistributed

from keras.models import Sequential

model2=Sequential()

model2.add(SimpleRNN(50,input_shape=(max_len+1,len(word2idx_2)),return_sequences=True))

model2.add(TimeDistributed(Dense(len(pos2idx),activation='softmax')))

model2.compile(loss="categorical_crossentropy",optimizer="adam")

model2.fit(input_data,target_data,batch_size=100,epochs=15)

for k2 in range(1,7):

    input_data=np.zeros((2000,max_len+1,len(word2idx_2)),dtype='float32')

    target_data=np.zeros((2000,max_len+1,len(pos2idx)),dtype='float32')

    for k,w in enumerate(X2[k2*2000:k2*2000+2000]):

        for k1,w1 in enumerate(w):

            input_data[k,k1,X2[k+k2*2000][k1]]=1

    for k,w in enumerate(X2[k2*2000:k2*2000+2000]):

        for k1,w1 in enumerate(w):

            target_data[k,k1,P2[k+k2*2000][k1]]=1

    model2.fit(input_data,target_data,batch_size=100,epochs=15)

input_data=np.zeros((586,max_len+1,len(word2idx_2)),dtype='float32')

target_data=np.zeros((586,max_len+1,len(pos2idx)),dtype='float32')

k2=7

for k,w in enumerate(X2[k2*2000:k2*2000+586]):

        for k1,w1 in enumerate(w):

            input_data[k,k1,X2[k+k2*2000][k1]]=1

for k,w in enumerate(X2[k2*2000:k2*2000+586]):

        for k1,w1 in enumerate(w):

            target_data[k,k1,P2[k+k2*2000][k1]]=1

model2.fit(input_data,target_data,batch_size=100,epochs=15)
word2idx_1 = {}

word_idx_1 = 0

for k,w in enumerate(tokens_pos_neg_11):

    for k1,w1 in enumerate(w):

        if w1 not in word2idx_1:

            word2idx_1[w1]=word_idx_1

            word_idx_1 += 1

print(word2idx_1)
len(word2idx_1)
Xtrain_1=[]

for k,w in enumerate(tokens_pos_neg_11):

    l=[]

    for k1,w1 in enumerate(w):

        #print(w1)

        #print(w1)

        l.append(word2idx_1[w1])

    Xtrain_1.append(l)
X1=[]

P1=[]

for x,z in zip(Xtrain_1,Ptrain):

    if len(x)==len(z):

        X1.append(x)

        P1.append(z)

        #selected_text_pos_neg_modified.append(s)
input_data=np.zeros((2000,max_len+1,len(word2idx_1)),dtype='float32')

target_data=np.zeros((2000,max_len+1,len(pos2idx)),dtype='float32')

for k,w in enumerate(X1[0:2000]):

    for k1,w1 in enumerate(w):

        input_data[k,k1,X1[k][k1]]=1

for k,w in enumerate(X1[0:2000]):

    for k1,w1 in enumerate(w):

        target_data[k,k1,P1[k][k1]]=1

from keras.layers import SimpleRNN,Dense,Activation,TimeDistributed

from keras.models import Sequential

model1=Sequential()

model1.add(SimpleRNN(50,input_shape=(max_len+1,len(word2idx_1)),return_sequences=True))

model1.add(TimeDistributed(Dense(len(pos2idx),activation='softmax')))

model1.compile(loss="categorical_crossentropy",optimizer="adam")

model1.fit(input_data,target_data,batch_size=100,epochs=15)

for k2 in range(1,7):

    input_data=np.zeros((2000,max_len+1,len(word2idx_1)),dtype='float32')

    target_data=np.zeros((2000,max_len+1,len(pos2idx)),dtype='float32')

    for k,w in enumerate(X1[k2*2000:k2*2000+2000]):

        for k1,w1 in enumerate(w):

            input_data[k,k1,X1[k+k2*2000][k1]]=1

    for k,w in enumerate(X1[k2*2000:k2*2000+2000]):

        for k1,w1 in enumerate(w):

            target_data[k,k1,P1[k+k2*2000][k1]]=1

    model1.fit(input_data,target_data,batch_size=100,epochs=15)

input_data=np.zeros((586,max_len+1,len(word2idx_1)),dtype='float32')

target_data=np.zeros((586,max_len+1,len(pos2idx)),dtype='float32')

k2=7

for k,w in enumerate(X1[k2*2000:k2*2000+586]):

        for k1,w1 in enumerate(w):

            input_data[k,k1,X1[k+k2*2000][k1]]=1

for k,w in enumerate(X1[k2*2000:k2*2000+586]):

        for k1,w1 in enumerate(w):

            target_data[k,k1,P1[k+k2*2000][k1]]=1

model1.fit(input_data,target_data,batch_size=100,epochs=15)
nlp = en_core_web_sm.load()

docs=[nlp(test_set.text[i]) for i in range(len(test_set.text))]

tokens = [[token.text.lower() for token in doc] for doc in docs]

a=y2_predicted == 'negative' 

b=y2_predicted == 'positive'

A=[]

for x,y in zip(a,b):

    A.append(x or y)

tokens_pos_neg_out=[]



for i,x in enumerate(A):

    if x:

        tokens_pos_neg_out.append(tokens[i])

for k,w in enumerate(tokens_pos_neg_out):

    w.insert(0,'#')

    w.append('#')
for k,w in enumerate(tokens_pos_neg_out):

    for k1,w1 in enumerate(w):

        if w1 not in word2idx:

            word2idx[w1]=word_idx

            word_idx += 1
Xtest=[]

for k,w in enumerate(tokens_pos_neg_out):

    l=[]

    for k1,w1 in enumerate(w):

        #print(w1)

        #print(w1)

        l.append(word2idx[w1])

    Xtest.append(l)
Xtest
tokens_pos_neg_out_3=[]

for k,w in enumerate(tokens_pos_neg_out):

    a=[]

    for k1,w1 in enumerate(w):

        a.append(w1[:3])

    tokens_pos_neg_out_3.append(a)
tokens_pos_neg_out_3
for k,w in enumerate(tokens_pos_neg_out_3):

    for k1,w1 in enumerate(w):

        if w1 not in word2idx_3:

            word2idx_3[w1]=word_idx_3

            word_idx_3 += 1
Xtest_3=[]

for k,w in enumerate(tokens_pos_neg_out_3):

    l=[]

    for k1,w1 in enumerate(w):

        #print(w1)

        #print(w1)

        l.append(word2idx_3[w1])

    Xtest_3.append(l)
Xtest_3
tokens_pos_neg_out_2=[]

for k,w in enumerate(tokens_pos_neg_out):

    a=[]

    for k1,w1 in enumerate(w):

        a.append(w1[:2])

    tokens_pos_neg_out_2.append(a)
for k,w in enumerate(tokens_pos_neg_out_2):

    for k1,w1 in enumerate(w):

        if w1 not in word2idx_2:

            word2idx_2[w1]=word_idx_2

            word_idx_2 += 1
Xtest_2=[]

for k,w in enumerate(tokens_pos_neg_out_2):

    l=[]

    for k1,w1 in enumerate(w):

        #print(w1)

        #print(w1)

        l.append(word2idx_2[w1])

    Xtest_2.append(l)
Xtest_2
tokens_pos_neg_out_1=[]

for k,w in enumerate(tokens_pos_neg_out):

    a=[]

    for k1,w1 in enumerate(w):

        a.append(w1[:1])

    tokens_pos_neg_out_1.append(a)
for k,w in enumerate(tokens_pos_neg_out_1):

    for k1,w1 in enumerate(w):

        if w1 not in word2idx_1:

            word2idx_1[w1]=word_idx_1

            word_idx_1 += 1
Xtest_1=[]

for k,w in enumerate(tokens_pos_neg_out_1):

    l=[]

    for k1,w1 in enumerate(w):

        #print(w1)

        #print(w1)

        l.append(word2idx_1[w1])

    Xtest_1.append(l)
Xtest_1
Xtest
cond=[]

for k,w in enumerate(Xtest):

    if max(w)<19598:

        cond.append(1)

    else:

        cond.append(0)
cond


for k,w in enumerate(Xtest_3):

    if max(w)<4320 and cond[k]!=1:

        cond[k]=2

    
sum(np.array(cond)==2)+sum(np.array(cond)==1)+sum(np.array(cond)==3)+sum(np.array(cond)==4)
for k,w in enumerate(Xtest_2):

    if max(w)<1046 and cond[k]!=1 and cond[k]!=2 :

        cond[k]=3
for k,w in enumerate(Xtest_1):

    if max(w)<71 and cond[k]!=1 and cond[k]!=2 and cond[k]!=3:

        cond[k]=4
len(cond)
P_theta_total=[]

for i in range(len(cond)):

    if cond[i]==1:

        output_seq=np.zeros((1,max_len+1,19598))

        for k,w in enumerate(Xtest[i]):

            output_seq[0,k,w]=1

        probs=model.predict_proba(output_seq,verbose=0)

        P_theta=[]

        for i in range(len(Xtest[i])):

            P_theta.append(list(probs[:,i,:][0]).index(max(probs[:,i,:][0])))

            

        P_theta_total.append(P_theta)

    elif cond[i]==2:

        output_seq=np.zeros((1,max_len+1,4320))

        for k,w in enumerate(Xtest_3[i]):

            output_seq[0,k,w]=1

        probs=model3.predict_proba(output_seq,verbose=0)

        P_theta=[]

        for i in range(len(Xtest_3[i])):

            P_theta.append(list(probs[:,i,:][0]).index(max(probs[:,i,:][0])))

            

        P_theta_total.append(P_theta)

    elif cond[i]==3:

        output_seq=np.zeros((1,max_len+1,1046))

        for k,w in enumerate(Xtest_2[i]):

            output_seq[0,k,w]=1

        probs=model2.predict_proba(output_seq,verbose=0)

        P_theta=[]

        for i in range(len(Xtest_2[i])):

            P_theta.append(list(probs[:,i,:][0]).index(max(probs[:,i,:][0])))

            

        P_theta_total.append(P_theta)

    elif cond[i]==4:

        output_seq=np.zeros((1,max_len+1,71))

        for k,w in enumerate(Xtest_1[i]):

            output_seq[0,k,w]=1

        probs=model1.predict_proba(output_seq,verbose=0)

        P_theta=[]

        for i in range(len(Xtest_1[i])):

            P_theta.append(list(probs[:,i,:][0]).index(max(probs[:,i,:][0])))

            

        P_theta_total.append(P_theta)
np.array(P_theta_total[3])>0
Xtest
list(np.array(tokens_pos_neg_out[3])[np.array(P_theta_total[3])>0])
L=[]

for i in range(len(tokens_pos_neg_out)):

    L.append(list(np.array(tokens_pos_neg_out[i])[np.array(P_theta_total[i])>0]))
len(L)
' '.join(L[3])
selected_text_predicted_pos_neg=[' '.join(str) for str in L]
selected_text_predicted_pos_neg
test_set.text[0]
sum(y2_predicted=='neutral')+len(L)
len(y2_predicted)
LL=[]

k=0

for i in range(len(y2_predicted)):

    if y2_predicted[i]=='neutral':

        LL.append(test_set.text[i])

    else:

        LL.append(selected_text_predicted_pos_neg[k])

        k=k+1
LL
submission=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

#`submission["selected_text"]=LL
#submission.to_csv('submission.csv', index=False)
data = {'test_test':  test_set.text,

        'extracted_text': LL,

        'sentiment_predicted': y2_predicted

        

        }
results = pd.DataFrame (data)
LL
X1=pd.DataFrame(X1_txt.toarray(), columns=vect1.get_feature_names())
prob = log_reg1.predict_proba(X1)
X2=pd.DataFrame(X2_txt.toarray(), columns=vect1.get_feature_names())
feature_indexes_2=[]

for row in X2.itertuples():

    feature_indexes_2.append(list(np.where(row==np.ones(len(row)))[0]))
feature_indexes_2=[]

for row in X2.itertuples():

    feature_indexes_2.append(list(np.where(row==np.ones(len(row)))[0]))
feature_indexes_2
feature_indexes_rectified_2=[]

for x in feature_indexes_2:

    x=np.array(x)-1

    feature_indexes_rectified_2.append(list(x))
feature_indexes_rectified_2
features_2=[]

for k,w in enumerate(feature_indexes_rectified_2):

    if len(w)>0:

        features_2.append(list(np.array(vect1.get_feature_names())[w]))

    else:

        features_2.append([])
features_2
a=list(log_reg1.coef_[0])
b=list(log_reg1.coef_[2])
e=zip(vect1.get_feature_names(),a,b)
f=[*e]
g={}

for w in f:

    g[w[0]]=[w[1],w[2]]
g
g['***']
values_neg=[]

values_pos=[]

for k,w in enumerate(features_2):

    a=[]

    b=[]

    for k1,w1 in enumerate(w):

        a.append(g[w1][0])

        b.append(g[w1][1])

    values_neg.append(a)

    values_pos.append(b)
np.array(values_neg[0])<0
list(np.array(features_2[0])[np.array(values_neg[0])>0])
tokens_positive=[]

tokens_negative=[]

for k,w in enumerate(features_2):

    tokens_negative.append(list(np.array(w)[np.array(values_neg[k])>0]))

    tokens_positive.append(list(np.array(w)[np.array(values_pos[k])>0]))
L1=[]

L2=[]

for w,w1 in zip(tokens_negative,tokens_positive):

    L1.append(' '.join(w) )

    L2.append(' '.join(w1) )
L1
for i in range(len(LL)):

    if len(LL[i])==0 and y2_predicted[i]=='negative':

        LL[i]=L1[i]

    elif len(LL[i])==0 and y2_predicted[i]=='positive':

        LL[i]=L2[i]

    elif len(LL[i])==0:

        LL[i]='_'
LL
y2_predicted[1]
submission["selected_text"]=LL
submission.to_csv('submission.csv', index=False)
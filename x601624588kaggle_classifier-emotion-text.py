import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



dat=[]

import os

# load emotion text

for page in range(3):

    dat.append(pd.read_excel('../input/emotion-text/emotion_text.xlsx',sheet_name = page))

               

Dat = pd.concat(dat)



Dat # to show
tag,tags = [],[]

for txt in Dat[['标签']].values.tolist():

    tags.extend(txt[0].replace('的','').split('，'))

    tag.append(txt[0].replace('的','').split('，'))

    

tag_dict = {t:i for i,t in enumerate(list(set(tags)))}



# to get tag vectors for each sentence, thus , it's a matrix



tag_mat = np.zeros((len(Dat),len(tag_dict)),dtype='int8')# init tag_vec to save 0/1 matrix

for i,t in enumerate(tag):

    for ti in t:

        tag_mat[i,tag_dict[ti]]=1

tag_mat=np.array(tag_mat)
# these tags only appear ones !! no more information



[list(tag_dict.keys())[i] for i,k in enumerate(np.array(tag_mat).sum(0)) if k<=1]
# glance the corelation among labels

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(6,5))

cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)

pt = pd.DataFrame(np.array(tag_mat)).corr()   

sns.heatmap(pt, linewidths = 0.05, vmax=1, vmin=-1,cmap="RdBu")

plt.show()
# tag is uncool for classification, thus, we need enbalance it quantity later

{list(tag_dict.keys())[i]:k for i,k in enumerate(np.array(tag_mat).sum(0)) if k>0}
# load stopword

with open('../input/stopwordch/stopwordCh.txt','r',encoding='utf8') as f:

    stopword = [w.strip() for w in f.readlines()]

    stopword.append(' ')
import jieba

jieba.load_userdict(['../input/emotion-word/anger.txt','../input/emotion-word/neg.txt','../input/emotion-word/pos.txt'])

s_lst = Dat[['内容']].values.tolist()# load sentence list



s_cut=[]# save sentence word cut

for s in s_lst:

    lst = jieba.cut(s[0])

    s_cut.append([w for w in list(lst) if w not in stopword+['\n','\xa0']])
# for strengthen the sentence

!pip install synonyms
import synonyms,random

# shuffle sample 



sample = list(zip(s_cut,tag_mat))

random.shuffle(sample)

s_cut,tag_mat = tuple(zip(*sample))



# divide sample into training set and test set at rate 0.8:0.2



s_cut_train = s_cut[:int(len(s_cut)*0.8)]

s_cut_test = s_cut[int(len(s_cut)*0.8):]

tag_mat_train = tag_mat[:int(len(tag_mat)*0.8)]

tag_mat_test = tag_mat[int(len(tag_mat)*0.8):]



# enhance the training set 

s_cut_train_extend = []

alpha = 0.9

beta = 0.9

gamma = 0.9

tag_mat_train_new = []

ind_cnt={list(tag_dict.values())[i]:k for i,k in enumerate(np.array(tag_mat).sum(0)) if k>0}



for j,s in enumerate(s_cut_train):

    # to add how many word seq for each one and make all kinds of quantity balance

    for k in range(int(sum(ind_cnt.values())/ind_cnt[tag_mat_train[j].argmax()])): 

        # form the taget vector at the same time

        tag_mat_train_new.append(tag_mat_train[j])

        

        s_new = []

        for i,w in enumerate(s): #choose one opption with rate

            

            # rexchange position with previous word at rate 1-beta

            if random.choices((0,1),(beta,1-beta))[0]&len(s_new):# opption at p = beta and isnot the first word

                nextword = s_new.pop()

                # invert

                s_new.append(w)

                s_new.append(nextword)

            # delet word with probability at 1-alpha

            elif random.choices((0,1),(alpha,1-alpha))[0]:# opption at p = alpha

                pass   

            # replace with synonyms at rate 1-gamma

            elif random.choices((0,1),(alpha,1-alpha))[0]:# opption at p = alpha

                if synonyms.nearby(w,2)[0]:

                    s_new.append(synonyms.nearby(w,2)[0][1])

            else:

                s_new.append(w)

            

            



        s_cut_train_extend.append(s_new)# add nearby sentences for ten times



# shuffle training set then       

sample = list(zip(s_cut_train_extend,tag_mat_train_new))

random.shuffle(sample)

s_cut_train_extend,tag_train = tuple(zip(*sample))



tag_train = np.array(tag_train)# tag vec that extend
np.array(tag_mat_train).sum(0)
np.array(tag_mat_train_new).sum(0)
word_lst=[]

for s in s_cut_train_extend+s_cut_test:

    word_lst.extend(s)

word_lst = list(set(word_lst))

d_w2i = {w:i for i,w in enumerate(word_lst)}

# save the dictionary to storage

with open('./d_w2i.txt','w') as f:

    f.write(str(d_w2i))
'''get digit data and organize'''

from keras.preprocessing.sequence import pad_sequences



LEN = 2000# the longest sequence

def tokenizer(d_w2i,txt_c,len=LEN):

    '''given dictionary word to digt , txt cut and pad LEN. return digt tok '''

    tok = []

    for i,s in enumerate(txt_c):

        tok.append([])

        for w in s:

            try:

                if d_w2i[w] != None:

                    tok[i].append(d_w2i[w])

                else:

                    tok[i].append(0)#if len of seq not approch to LEN add zeros before the indexs

            except:

                tok[i].append(0)

    tok = pad_sequences(tok,LEN)

    return tok



tok_train = tokenizer(d_w2i,s_cut_train_extend)

tok_test = tokenizer(d_w2i,s_cut_test)
'''load word embeding model'''

from keras.preprocessing.sequence import pad_sequences

import codecs

def Load_model_emb(data_dict,embMod_path):

    '''

    Input: word-index dictionary, Mode path;

    Return: word Embedding Matrix which can be process by keras.layers.Embedding

    '''

    with codecs.open(embMod_path,'r','utf-8') as f:

        emb_mat = np.zeros((len(data_dict) + 1, 300))

        size =None

        num = 0 #num of hits

        for line in f:

            line = line.strip().split(' ')

            if not size:

                size=line

                continue

            if line[0] in data_dict.keys():

                num+=1

                emb_mat[data_dict[line[0]]] = np.asarray(line[1:], dtype='float32')

    return emb_mat,len(data_dict)-num

Emb_mod,oov = Load_model_emb(d_w2i,'../input/w2v-cn/sgns.sogounews.bigram-char')

oov_rate = oov/len(d_w2i) # print num of word that out of vocabulary

oov_rate
'''build model'''

from keras.models import Sequential,save_model,load_model

from keras.layers import *

def build_mod(tok,tag,emb_mat,drop_rate=0.3):

    model = Sequential()

    # embedding with unknown method



    #Embedding(if cant work, we may input the vectors into the network directly)

    model.add(Embedding(emb_mat.shape[0],emb_mat.shape[1],weights=[emb_mat],input_length=tok.shape[1],trainable = False,))

    # drop some data randomly, thus, we can prevent overfitting

    model.add(Dropout(drop_rate))



    # GRU

    model.add(GRU(2**4))

    model.add(Dropout(drop_rate))

    

    # fully connection

    model.add(Dense(2**6,activation='relu'))

    model.add(Dropout(drop_rate))

    

    #the last layer, we use softmax for classification

    model.add(Dense(tag.shape[1],activation = 'softmax'))



    # set loss as crossentropy,and use method as Adam, and use accuracy as metric to analyse the result

    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

    return model

mod = build_mod(np.array(tok_train),np.array(tag_train),Emb_mod)

mod.summary()
mod.fit(x=tok_train,y=np.array(tag_train), validation_split=0.2,batch_size=2**4,epochs=6,workers=4)
mod.evaluate(tok_test,np.array(tag_mat_test))
show_data = (np.array(tag_mat_train).sum(0),np.array(tag_mat_train_new).sum(0),np.array(tag_mat_test).sum(0))

pd.DataFrame(data=show_data,columns=tag_dict.keys(),index=['训练集标签数','增强的训练集标签数目','测试集标签数目'])
import jieba

# add emotional words to vocabulary

jieba.load_userdict(['../input/emotion-word/anger.txt','../input/emotion-word/neg.txt','../input/emotion-word/pos.txt'])



# load stopword

with open('../input/stopwordch/stopwordCh.txt','r',encoding='utf8') as f:

    stopword = [w.strip() for w in f.readlines()]

    stopword.append(' ')

    

s_lst = input('please input sentence')

if type(s_lst) is str: # format Str to List

    s_lst = [s_lst]



# save sentence word cut

s_cut=[]

for s in s_lst:

    lst = jieba.cut(s)

    s_cut.append([w for w in list(lst) if w not in stopword+['\n','\xa0']])

    

# load dictionary 

with open('./d_w2i.txt','r') as f:

    d_w2i = eval(f.read())



# get digit label matrix

'''get digit data and organize'''

from keras.preprocessing.sequence import pad_sequences



LEN = 2000# the longest sequence

def tokenizer(d_w2i,txt_c,len=LEN):

    '''given dictionary word to digt , txt cut and pad LEN. return digt tok '''

    tok = []

    for i,s in enumerate(txt_c):

        tok.append([])

        for w in s:

            try:

                if d_w2i[w] != None:

                    tok[i].append(d_w2i[w])

                else:

                    tok[i].append(0)#if len of seq not approch to LEN add zeros before the indexs

            except:

                tok[i].append(0)

    tok = pad_sequences(tok,LEN)

    return tok



tok = tokenizer(d_w2i,s_cut)



'''predict with model'''

from keras.models import load_model

res = mod.predict(tok)



'''afterward'''

pd.DataFrame(res/res.max(),columns=['愉快', '颓废', '激活', '厌烦', '满足', '宁静', '疲乏', '愤怒', '苦恼', '沮丧', '紧张', '悲观'])
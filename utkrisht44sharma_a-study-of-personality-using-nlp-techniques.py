import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
text=pd.read_csv("../input/mbti_1.csv")
posts=text.values.tolist()
mbti_list=['ENFJ','ENFP','ENTJ','ENTP','ESFJ','ESFP','ESTJ','ESTP','INFJ','INFP','INTJ','INTP','ISFJ','ISFP','ISFP','ISTP']
values = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
index = np.arange(len(mbti_list))
for post in posts:
    for i in range(0,len(mbti_list)):
        if post[0] == mbti_list[i]:
            values[i]=values[i]+1
plt.bar(index,values)
plt.xlabel('Personality Type',fontsize=3)
plt.ylabel('No of persons',fontsize=5)
plt.xticks(index,mbti_list,fontsize=8,rotation=35)
plt.title('Distribution of types among Dataset(1 person=50 tweets)')
plt.show()
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import re

def train_w2v_using_key(temp):# I'm too lazy to learn regex in python
    perlist=list()
    if temp=="SJ":
        for i in posts:
            if i[0]=='ISFJ' or i[0]=='ISTJ' or i[0]=='ESFJ' or i[0]=='ESTJ':
                perlist.append(i[1])
    if temp == 'SP':
        for i in posts:
            if i[0]=='ISFP' or i[0]=='ISTP' or i[0]=='ESFP' or i[0]=='ESTP':
                perlist.append(i[1])
    else:       
         for i in posts:
            if temp in i[0]:
                perlist.append(i[1])
    for i in range(0,len(perlist)): # using some code https://www.kaggle.com/prnvk05/rnn-mbti-predictor for filtering out links and numbers from the text 
        tempstr = ''.join(str(e) for e in perlist[i])
        post=tempstr.lower()
        post=post.replace('|||',"")
        post = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', post, flags=re.MULTILINE) 
        puncs1=['@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']','|','\\','"',"'",';',':','<','>','/']
        for punc in puncs1:
            post=post.replace(punc,'') 

        puncs2=[',','.','?','!','\n']
        for punc in puncs2:
            post=post.replace(punc,' ') 
        post=re.sub( '\s+', ' ', post ).strip()
    perlist[i]=post
    
    word_tokens=[]
    for i in range(0,len(perlist)): 
        word_tokens.append(word_tokenize(perlist[i]))

    model = Word2Vec(word_tokens, min_count=1)
    model.save(temp+".bin")
train_w2v_using_key("NT")
train_w2v_using_key("NF")
train_w2v_using_key("SP")
train_w2v_using_key("SJ")
model=Word2Vec.load("NT.bin")
model.wv.similarity("defend","justify")
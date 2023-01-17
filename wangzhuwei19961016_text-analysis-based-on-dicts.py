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
import codecs

from collections import defaultdict

import jieba

import xlrd
##分词，去除停用词

def seg_word(sentence):

    seg_list=jieba.cut(sentence)

    seg_result=[]

    for w in seg_list:

        seg_result.append(w)

    #读取停用词

    stopwords=set()

    fr=codecs.open('/kaggle/input/stopwords/stopwords.txt','r','utf-8')

    for word in fr:

        stopwords.add(word.strip())

    fr.close()

    return list(filter(lambda x:x not in stopwords,seg_result))

##分词结果进行分类，情感词，否定词，程度副词

def classify_words(word_list):

    sen_file=open('/kaggle/input/bosonnlp-sentiment-scoretxt/BosonNLP_sentiment_score.txt','r',encoding='utf-8')

    sen_list=sen_file.read().splitlines()

    sen_dict=defaultdict()

    for s in sen_list:

        if len(s.split(' '))==2:

            sen_dict[s.split(' ')[0]] = s.split(' ')[1]

            

##读取否定词

    not_word_file=open('/kaggle/input/degree/nodict.txt','r',encoding='utf-8')

    not_word_list=not_word_file.read().splitlines()

    

    #读取程度副词文件,程度副词数量较少，有能力可以自行建立更为强大的程度副词文件。

    degree_file=open('/kaggle/input/degree-1/degree.txt','r',encoding='GBK')

    degree_list=degree_file.readlines()

    degree_dic=defaultdict()

    for d in degree_list:

        for i in range(0,len(degree_list)):

            degree_dic[degree_list[i].split(",")[0]] = degree_list[i].split(",")[1]

        

    sen_word=dict()

    not_word=dict()

    degree_word=dict()

    for word in word_list:

        if word in sen_dict.keys() and word not in not_word_list and word not in degree_dic.keys():

            sen_word[word]=sen_dict[word]

        elif word in not_word_list and word not in degree_dic.keys():

            not_word[word]=-1

        elif word in degree_dic.keys():

            degree_word[word]=degree_dic[word]

    sen_file.close()

    degree_file.close()

    not_word_file.close()

    #print(sen_word)

    return sen_word,not_word,degree_word
##计算得分，再相加

def score_sentiment(sen_word,not_word,degree_word,seg_result):

    w=1

    score=0

    ##遍历分词结果，根据程度副词和否定词调整权重，并提取情感词的赋值大小与权重相乘，并返回总得分

    for i in range(0,len(seg_result)):

        if seg_result[i] in degree_word.keys():

            w*=float(degree_word[seg_result[i]])

        elif seg_result[i] in not_word.keys():

            w*=-1

        elif seg_result[i] in sen_word.keys():

            score+=float(w)*float(sen_word[seg_result[i]])

            w=1

        return score

    
def sentiment_score(sentence):

    seg_list=seg_word(sentence)

    sen_word,not_word,degree_word=classify_words(seg_list)

    #print(seg_list)

    score=score_sentiment(sen_word,not_word,degree_word,seg_list)

    return score

if __name__=='__main__':

    score=sentiment_score('喜欢地地道道')

    print(score)
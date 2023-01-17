# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from time import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import re

from sklearn.metrics import log_loss



from nltk.corpus import stopwords

from nltk import SnowballStemmer
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
# Check for any null values

print(df_train.isnull().sum())

print(df_test.isnull().sum())
# Add the string 'empty' to empty strings

train = df_train.fillna('')

test = df_test.fillna('')
pal = sns.color_palette()

data=pd.DataFrame()

stops = set(stopwords.words("english"))
def drop_stopwords(tokens):

    final_tokens=set()

    for word in tokens:

        if word not in stops:

            final_tokens.add(word)

    return final_tokens





def zzbds(sent):

    sent=sent.lower()

    # 正则表达式去标点

    tokens=[n for n in re.findall(r'[^:?(),.!" ]+',sent)]

    tokens=drop_stopwords(tokens)

    return tokens



def find_noun(sent):

    tokens=[n for n in sent]

    tags=nltk.pos_tag(tokens)

    a=[b[0] for b in tags if b[1] in ['NN','NNS','NNP','NNPS']]



    return len(a)
def calculate_train_word_match(row):

    if len(row['token_q1']) == 0 & len(row['token_q2']) == 0:

        R = 0

    else:

        R=(row['count_nn1']+row['count_nn2'])

#        R = 2*len(row['common_words'])/(len(row['token_q1'])+len(row['token_q2']))

#    R = -(len(row['q1_un'])+len(row['q2_un']))/(len(row['common_words'])+1)

    return R
t0 = time()

data.loc[:,'id']=df_train.iloc[:,0]

data.loc[:,'is_duplicate']=df_train.iloc[:,5]

data.loc[:,'question1']=df_train.question1.apply(lambda x:str(x))

data.loc[:,'question2']=df_train.question2.apply(lambda x:str(x))

data.loc[:,'token_q1']=data.question1.apply(zzbds)

data.loc[:,'token_q2']=data.question2.apply(zzbds)

data.loc[:,'common_words']=data.apply(

        lambda x: set(x['token_q1']).intersection(set(x['token_q2'])),axis=1)

data.loc[:,'q1_un']=data.apply(

        lambda x:set(x['token_q1']).difference(set(x['common_words'])),axis=1)

data.loc[:,'q2_un']=data.apply(

        lambda x:set(x['token_q2']).difference(set(x['common_words'])),axis=1)

data.loc[:,'count_nn1']=data.q1_un.apply(lambda x:find_noun(x))

data.loc[:,'count_nn2']=data.q2_un.apply(lambda x:find_noun(x))



plt.figure(figsize=(15, 5))

train_word_match2 = data.apply(calculate_train_word_match, axis=1, raw=True)

plt.hist(train_word_match2[df_train['is_duplicate'] == 0].dropna(),

         bins=20, normed=True, label='Not Duplicate')

plt.hist(train_word_match2[df_train['is_duplicate'] == 1].dropna(),

         bins=20, normed=True, alpha=0.7, label='Duplicate')

plt.legend()

plt.title('Label distribution over word_match_share', fontsize=15)

plt.xlabel('word_match_share', fontsize=15)



print("training time:", round(time()-t0, 3), "s")



from sklearn.metrics import roc_auc_score

print('Original AUC:', roc_auc_score(

        df_train['is_duplicate'], train_word_match2))
data.info()
data2=data.iloc[:,7:]

data2.info()
data2.to_csv('data_train0405.csv', index=False)
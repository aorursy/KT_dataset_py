# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input","-l"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('../input/train.csv')

#df_train.head()

#df_test=pd.read_csv('../input/test.csv')
df_train = np.array(df_train)

print(df_train[0])

i=0

for line in df_train:

    #print(line)

    if line[5]==1:

        i+=1

print(i/len(df_train))

print("train length:",len(df_train))

allque_list=[]

for line in df_train:

    allque_list.append(line[3])

    allque_list.append(line[4])

print('que length:',len(allque_list))

print(allque_list[0],allque_list[1])

from sklearn.feature_extraction.text import TfidfVectorizer

  

tfidf_vectorizer = TfidfVectorizer(min_df = 2)

tfidf_matrix = tfidf_vectorizer.fit_transform(np.array(allque_list,dtype='U'))

print(tfidf_matrix[1:2])
from gensim.models import word2vec
model = word2vec.Word2Vec(tfidf_matrix,size=200)
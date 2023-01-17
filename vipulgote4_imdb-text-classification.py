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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import preprocessing
import re
import string
import nltk
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
df=pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df.head()
df=df.drop_duplicates()
df.groupby('sentiment').count()
df.dtypes
review_Array=df['review'].values.astype('str')
review_Array.shape
review_list=[]
for i in range(len(review_Array)):
    string2=review_Array[i]
    string2=string2.lower()
    string2=re.sub(r'\d+','',string2)
    string2=string2.translate(str.maketrans('','', string.punctuation))
    string2=sent_tokenize(string2)
    review_list.append(string2)
review_list
tokens = [word_tokenize(str(w)) for w in review_list]
porter=PorterStemmer()
word=[porter.stem(words) for words in tokens]
word
stop_Words=stopwords.words('english')
print(stop_Words)
words=[w for w in word if not w in stop_Words]
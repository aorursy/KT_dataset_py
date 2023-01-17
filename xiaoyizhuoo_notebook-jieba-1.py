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
import jieba
jieba.lcut("这是一句句子")
my_data = pd.read_excel('../input/2017xls/2017.xls')
from gensim.models import Word2Vec, KeyedVectors
import nltk
my_data.head(10)
target_title1 = my_data["合同名称"].values

target_title1
corpus_word = my_data["合同名称"].apply(jieba.lcut)

corpus = corpus_word.tolist()

corpus
model = Word2Vec(corpus,min_count=1,size=32)
vec = model["冷轧"]+model["电器"]+model["产品"]
model.most_similar([vec])
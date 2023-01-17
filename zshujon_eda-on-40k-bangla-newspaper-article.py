# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pickle
from collections import Counter

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
with open('../input/40k_bangla_newspaper_article.p', 'rb') as f:
    data = pickle.load(f)
print(type(data))
data_len = len(data)
print(data_len)
print(data[0])
category_dist = {}
txt = ''
for i in data:
    txt += i['content'] + ' '
    if i['category'] not in category_dist:
        category_dist[i['category']] = 1
    else:
        category_dist[i['category']] += 1
category_dist
len(txt.split(' '))
word_dist = Counter(txt.split(' '))


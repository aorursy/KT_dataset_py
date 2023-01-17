# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

df = pd.read_csv('../input/class-descriptions.csv')
df.head()

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

df['english'] = df['description'].apply(isEnglish)
xdf = df[df.english == False]
print('shape: ', xdf.shape)
xdf.head()
strange_label_list = xdf['label_code'].tolist()
x = pd.read_csv('../input/classes-trainable.csv')
for l in list(x.label_code.unique()):
    if l in strange_label_list:
        print('Got {} in trainable labels'.format(l))
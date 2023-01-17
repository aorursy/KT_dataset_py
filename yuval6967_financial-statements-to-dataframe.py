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
import json

from io import StringIO



with open('../input/financial-statement-extracts/2016q1.json') as json_file:

    data = json.load(json_file)
list(data.keys())
df_num = pd.read_csv(StringIO(data['num.txt']),sep='\t')

print(df_num.shape)

df_num.head()
df_pre = pd.read_csv(StringIO(data['pre.txt']),sep='\t')

print(df_pre.shape)

df_pre.head()
df_sub = pd.read_csv(StringIO(data['sub.txt']),sep='\t')

print(df_sub.shape)

df_sub.head()
df_tag = pd.read_csv(StringIO(data['tag.txt']),sep='\t')

print(df_tag.shape)

df_tag.head()
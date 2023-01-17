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
import re
group_df = pd.read_csv('/kaggle/input/undrg-rd1-listings/Extra Material 2 - keyword list_with substring.csv')

name_df = pd.read_csv('/kaggle/input/undrg-rd1-listings/Keyword_spam_question.csv')
group_df.head()
group_df.shape
name_df.shape
from collections import defaultdict
dic = defaultdict(list)

for group, keywords in zip(list(group_df['Group'])[::-1],list(group_df['Keywords'])[::-1]):

    for keyword in keywords.split(", "):

        dic[group].append(keyword)

        dic[group].sort(reverse=True)

print(dic)
dic2 ={}

for key in dic:

    for ls in dic[key]:

        dic2[ls]=key

print(dic2)
names = list(name_df["name"])

names = [re.sub(r'[^\x00-\x7f]',r'',n) for n in names]

names = [n.lower() for n in names]
names
PERMITTED_CHARS ="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "

names = ["".join(c for c in n if c in PERMITTED_CHARS) for n in names]
names
res =[]

for name in names[:]:

    group_list =[]

    product_list=[]

    for key in dic2:

        if key in name:

            if(key not in product for product in product_list):

                group_list.append(dic2[key])

    group_list.sort(reverse=False)

    res.append(group_list)
df_res = pd.DataFrame({"index":range(len(res)),"groups_found":res})

df_res.to_csv("submission.csv",index=False)

pd.read_csv("submission.csv")
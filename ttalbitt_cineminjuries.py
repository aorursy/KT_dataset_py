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
df = pd.read_csv('/kaggle/input/all-injuries-in-cinematography-19142019/movie_injury.csv')
df.shape
import seaborn as sns
df['DescSplit'] = df['Description'].apply(lambda x : x.split())
df['DescLength'] = df['DescSplit'].apply(lambda x : len(x)) 
sns.distplot(df['Year'])
from nltk.corpus import stopwords

import re



eng_stops = stopwords.words('english')

regex = re.compile('[^a-zA-Z0-9]+')
word_bank = {}

for d in df.DescSplit:

    for w in d:

        c = regex.sub('', w).lower()

        if c not in eng_stops:

            g = word_bank.get(c)

            word_bank.update({c : 1 if g is None else g + 1})
sorted(word_bank.items(), key=lambda x:x[1], reverse=True)
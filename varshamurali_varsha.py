# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/dropped_required_stats.csv",low_memory=False)

df = df.drop(['smart_201_raw'], axis=1)
df = df[df['smart_5_raw'] != 'smart_5_raw']

df.head()

df['capacity_bytes']=df['capacity_bytes'].astype(np.int64)

df['failure']=df['failure'].astype(np.int64)

columns = ['smart_5_raw','smart_10_raw','smart_12_raw','smart_184_raw','smart_187_raw','smart_188_raw','smart_189_raw','smart_190_raw','smart_196_raw','smart_197_raw','smart_198_raw','smart_199_raw','smart_200_raw']

df['date'] =  pd.to_datetime(df['date'])





for col in columns:

        df[col] = df[col].astype(np.float64)
df.dtypes
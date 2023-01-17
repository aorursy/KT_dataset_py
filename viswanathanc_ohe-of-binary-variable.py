# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.DataFrame(np.random.randint(0,2,100),columns=['bin'])
df.head()
from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse=False) #OneHotEncoder by default returns CSR format

df2=ohe.fit_transform(df)
df2
df.to_numpy().T==df2[:,1]
df.to_numpy().T==df2[:,0]
pd.get_dummies(df,columns=['bin'],drop_first='True')
import os

import warnings

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import math as mt

import scipy



from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nRowsRead = 1000 # specify 'None' if want to read whole file

# free1.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/luteinizing-hormone-in-blood-samples/lh.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'lh.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
sns.pairplot(df1)
ax = sns.scatterplot(x="time", y="value", data=df1)
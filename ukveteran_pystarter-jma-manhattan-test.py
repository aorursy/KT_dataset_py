from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import seaborn as sns

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nRowsRead = 1000 # specify 'None' if want to read whole file

# free1.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/manhattan-test/manhattan_Test.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'manhattan_Test.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
p = df1.hist(figsize = (20,20))
f, ax = plt.subplots(figsize=(8, 6))

sns.countplot(x="Landmark", data=df1, color="c")

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.countplot(x="HistoricDistrict", data=df1, color="c")

plt.show()
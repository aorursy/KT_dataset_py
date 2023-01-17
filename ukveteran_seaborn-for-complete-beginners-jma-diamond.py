import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cufflinks as cf

import sklearn

from sklearn import svm,preprocessing

import seaborn as sns

import plotly.graph_objs as go

import plotly

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

import os
import pandas as pd

df= pd.read_csv("../input/diamond-price-prediction/diamonds.csv",index_col=0)
df.head()
df.columns
df.info()
df.describe()
f, ax = plt.subplots(figsize=(8,6))

x = df['carat']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = df['depth']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = df['price']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = df['carat']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = df['depth']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = df['price']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = df['carat']

x = pd.Series(x, name="Carat")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = df['depth']

x = pd.Series(x, name="Depth")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = df['price']

x = pd.Series(x, name="Price")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="carat", y="depth", data=df)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="carat", y="price", data=df)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="depth", y="price", data=df)

plt.show()
g = sns.JointGrid(x="carat", y="depth", data=df)

g = g.plot(sns.regplot, sns.distplot)
g = sns.JointGrid(x="carat", y="price", data=df)

g = g.plot(sns.regplot, sns.distplot)
g = sns.JointGrid(x="depth", y="price", data=df)

g = g.plot(sns.regplot, sns.distplot)
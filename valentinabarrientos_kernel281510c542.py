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
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import squarify

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import base64

import io

from scipy.misc import imread

import codecs

from IPython.display import HTML

from matplotlib_venn import venn2

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

response=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv',encoding='ISO-8859-1')
response.head()

print('El número total de encuestados:',response.shape[0])

print('Número total de barrios encuestados:',response['neighbourhood'].nunique())
resp_coun=response['neighbourhood'].value_counts()[:15].to_frame()

sns.barplot(resp_coun['neighbourhood'],resp_coun.index,palette='inferno')

plt.title('Top 15 barrios por número de encuestados')

plt.xlabel('')

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.show()

tree=response['neighbourhood'].value_counts().to_frame()

squarify.plot(sizes=tree['neighbourhood'].values,label=tree.index,color=sns.color_palette('RdYlGn_r',52))

plt.rcParams.update({'font.size':20})

fig=plt.gcf()

fig.set_size_inches(40,15)

plt.show()
plt.subplots(figsize=(22,12))

sns.countplot(y=response['neighbourhood_group'],order=response['neighbourhood_group'].value_counts().index)

plt.show()
plt.subplots(figsize=(50,10))

response['price'].hist(bins=40,edgecolor='black')

plt.xticks(list(range(0,1000,150)))

plt.title('Distribucion por precios')

plt.show()
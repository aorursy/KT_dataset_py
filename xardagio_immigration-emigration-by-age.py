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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from plotly import __version__

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

%matplotlib inline
data = pd.read_csv('../input/immigrants_emigrants_by_age.csv')
data.head()
data.info()
data[['Immigrants','Emigrants']].describe()
sum(data['Immigrants'])
sum(data['Emigrants'])
plt.figure(figsize=(15,5))

sns.barplot(data=data[['Immigrants','Emigrants']],estimator=sum)
plt.figure(figsize=(15,5))

plt.title('2015-2017 Immigrants by age')

sns.barplot(x=data['Age'],y=data['Immigrants'],estimator=sum,hue=data['Year'])
plt.figure(figsize=(15,5))

plt.title('2015-2017 Emigrants by age')

sns.barplot(x=data['Age'],y=data['Emigrants'],estimator=sum,hue=data['Year'])
dist_full_data = data[['Age','District Name','Immigrants','Emigrants']][data['District Name']!='No consta']

dist_full_data = dist_full_data.set_index(['District Name','Age'])

dist_full_data = dist_full_data.groupby(level=[0,1]).sum()

dist_full_data.rename(index={'0-4':'00-04','5-9':'05-09'}, inplace=True)

dist_full_data = dist_full_data.sort_index()
layout = dict(title='2015-2017 Immigrants/Emigrants by age and district',geo=dict(showframe=False))

dist_full_data.iplot(kind='bar',layout=layout)
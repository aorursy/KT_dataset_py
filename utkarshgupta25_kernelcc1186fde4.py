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
df = pd.read_csv('../input/seeds_dataset.txt', sep = '\t', header = None)



df.head()
df.columns = ['Area', 'Perimeter', 'Compactness', 'Kernel_length', 'Kernel_width', 'Asymmetry', 'Groove_length', 'Class']



df.head()
import seaborn as sns

import matplotlib.pyplot as plt



def plotBarChart(data, col, label):

    g = sns.FacetGrid(data, col = col)

    g.map(plt.hist, label, bins = 10)

    

for val in df.columns[:-1]:

    plotBarChart(df, 'Class', val)
df1 = df[df['Class'] != 3]



df1.head()
from sklearn.model_selection import train_test_split as tts

from sklearn.linear_model import LogisticRegression as lr



x = df1.drop(columns = ['Asymmetry' ,'Class'])

y = df1['Class']



x_train, x_test, y_train, y_test = tts(x,y)



regressor = lr()



regressor.fit(x_train, y_train)



regressor.score(x_test, y_test)
x = df.drop(columns = ['Asymmetry', 'Class'])

y = df['Class']



x_train, x_test, y_train, y_test = tts(x,y)



regressor = lr()



regressor.fit(x_train, y_train)



regressor.score(x_test, y_test)
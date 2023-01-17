import numpy as np

import pandas as pd

import random

import missingno as msno 

from sklearn.model_selection import KFold

import lightgbm as lgb

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

df.head()
df['SARS-Cov-2 exam result'] = [0 if a == 'negative' else 1 for a in df['SARS-Cov-2 exam result'].values]



Y = df['SARS-Cov-2 exam result']



df = df.drop([

    "Patient ID",

    'Patient addmited to regular ward (1=yes, 0=no)',

    'Patient addmited to semi-intensive unit (1=yes, 0=no)',

    'Patient addmited to intensive care unit (1=yes, 0=no)'

], axis=1)
df.head()
df.describe(include='all') # Show data descriptive summary
df = df.fillna(df.mean())
df.head(5)
df.corr()
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(20, 10))

matrix = np.triu(df.corr())

sns.heatmap(df.corr(), mask=matrix)
df.count()
plt.style.use('ggplot')

plt.figure(figsize=(8,8))

df['SARS-Cov-2 exam result'].value_counts().plot(kind='bar')

plt.title('Label Distribution')

plt.show()
df['SARS-Cov-2 exam result'].value_counts()
msno.bar(df) 
msno.heatmap(df) 
msno.matrix(df)
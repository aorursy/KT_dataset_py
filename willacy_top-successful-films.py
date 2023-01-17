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
df = pd.read_csv("/kaggle/input/marvel-vs-dc/db.csv", encoding = "ISO-8859-1", index_col=0)

df.head()
df.index -= 1

df.shape
df.head()
df.columns = [x.replace(" ", "_") for x in df.columns]

df = df.rename(columns={df.iloc[:,8].name:'Gross_USA'})

df.Budget = df.Budget.astype('int64')
df['Profit_in_%'] = (df['Gross_Worldwide']/df['Budget']*100).astype('int64')
top_features = ['Rate','Metascore','Gross_Worldwide','Profit_in_%']

films = {}

for i in top_features:

    for h, j in enumerate(df.sort_values(by=i, ascending=False).index):

        if j in films:

            films[j] += h

        else:

            films[j] = h



df['Score'] = df.index.map(films)
df.sort_values(by='Score').head(10)
df.sort_values(by='Score', ascending=False).head(10)
df['Score2'] = 0

for i in top_features:

    df['Score2'] += df[i].rank(ascending=False)
df['Score'] = df['Score'].rank()

df['Score2'] = df['Score2'].rank()
df.sort_values(by='Score2').head(10)
df.sort_values(by='Score2', ascending=False).head(10)
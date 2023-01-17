# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/stackindex/MLTollsStackOverflow.csv')
df.info()
df.describe()
df.dtypes
df.columns
df.isnull().sum()
df.shape
df.mode
df.Tableau.mode
df["Tableau"].fillna(value = 0.0, inplace = True)
df.Tableau
import matplotlib.pyplot as plt

import seaborn as sns
fig , ax = plt.subplots(figsize=(20,15))

plt.plot(df['python'], df['r'], df['pandas'], df['numpy'])
labels = 'Python' , 'NLTK', 'Pandas', 'Seaborn'

sizes = [df.loc[106, 'python'], df.loc[106, 'numpy'], df.loc[106, 'pandas'], df.loc[106, 'seaborn']]

colors = ['blue', 'red', 'green', 'black']



plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow = False)

centre_circle = plt.Circle((0,0),0.25,color='black', fc='white',linewidth=1.25)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.axis('equal')

plt.title('SOF on 17-Nov for Python and Python Packages')

plt.show()  
labels = 'Python' , 'NLTK', 'Pandas', 'Seaborn'

sizes = [df.loc[106, 'python'], df.loc[106, 'numpy'], df.loc[106, 'pandas'], df.loc[106, 'seaborn']]

colors = ['blue', 'red', 'green', 'black']



plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow = False)



plt.axis('equal')

plt.title('SOF on 17-Nov for Python and Python Packages')

plt.show()  
df['year'] = df['month'].apply( lambda x : x[0:2])
df['year'].head(5)
df.year.mode
df['year'].isnull().any()
df.columns
df.plot()
df.plot.area()
df.plot.scatter(x = 'python', y = 'year')
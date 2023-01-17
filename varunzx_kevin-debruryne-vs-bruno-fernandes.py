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
import seaborn as sns

import matplotlib.pyplot as plt



players=pd.read_csv('/kaggle/input/epl-stats-20192020/players_1920_fin.csv')

players.head()
players.info()
debru=players[players['full']=='Kevin De Bruyne']

bruno=players[players['full']=='Bruno Miguel Borges Fernandes']
plt.subplot(1,2,1)

sns.countplot(x='assists',data=debru).set_title('DE BRUYNE') # More on color options later

plt.subplot(1,2,2)

sns.countplot(x='assists',data=bruno).set_title('BRUNO FERNANDES')
plt.subplot(1,2,1)

sns.boxplot(debru['assists'].value_counts()/38).set_title('DE BRUYNE')

plt.subplot(1,2,2)

sns.boxplot(bruno['assists'].value_counts()/14).set_title('BRUNO FERNANDES')
plt.subplot(1,2,1)

sns.countplot(x='bonus',data=debru).set_title('DE BRUYNE') # More on color options later

plt.subplot(1,2,2)

sns.countplot(x='bonus',data=bruno).set_title('BRUNO FERNANDES')
sns.boxplot(debru['bonus'].value_counts()/38)
sns.boxplot(bruno['bonus'].value_counts()/14)


g = sns.FacetGrid(data=debru,col='was_home')

g.map(plt.hist,'minutes')



g = sns.FacetGrid(data=bruno,col='was_home')

g.map(plt.hist,'minutes')
plt.subplot(1,2,1)

sns.countplot(x='assists',data=debru,hue='was_home').set_title('DE BRUYNE') # More on color options later

plt.subplot(1,2,2)

sns.countplot(x='assists',data=bruno,hue='was_home').set_title('BRUNO FERNANDES')
plt.subplot(1,2,1)

sns.countplot(x='goals_scored',data=debru,hue='was_home').set_title('DE BRUYNE')

plt.subplot(1,2,2)

sns.countplot(x='goals_scored',data=bruno,hue='was_home').set_title('BRUNO FERNANDES')
plt.subplot(1,2,1)

sns.boxplot(debru['goals_scored'].value_counts()/38).set_title('DE BRUYNE')

plt.subplot(1,2,2)

sns.boxplot(bruno['goals_scored'].value_counts()/14).set_title('BRUNO FERNANDES')
debru['opponent_team'].unique()
bruno['opponent_team'].unique()
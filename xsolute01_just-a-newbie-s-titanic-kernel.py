import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
print('I have: ' + str(os.listdir("../input")))
data = pd.read_csv('../input/train.csv')

data.head()
data.isnull().sum()
f, ax = plt.subplots()

sns.countplot('Survived', data=data, ax=ax)

plt.show()
f, ax = plt.subplots(figsize=(5, 5.5))

data['Survived'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax,shadow=True)
ax.set_title('Survived')
ax.set_ylabel('')

plt.show()
data.groupby(['Sex', 'Survived'])['Survived'].count()
f, ax = plt.subplots()
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax)
ax.set_title('The Relationship between Survived and Sex')
plt.show()
f, ax = plt.subplots()
sns.countplot('Sex', hue='Survived', data=data, ax=ax)
ax.set_title('Survived and Dead with Sex')
plt.show()
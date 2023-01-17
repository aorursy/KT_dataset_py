# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.



#plt.style.use('fivethirtyeight')

%matplotlib inline
X = pd.read_csv('/kaggle/input/train.csv')
current_year = now = datetime.datetime.now().year 

X.loc[X['HouseYear'] > current_year, 'HouseYear'] = current_year
X['HouseYear'].value_counts().reset_index
plt.figure(figsize = (16, 8))



plt.subplot(121)

X['HouseYear'].hist(density=True)  

plt.ylabel('count')

plt.xlabel('HouseYear')



plt.subplot(122)

sns.kdeplot(X['HouseYear'], shade=True, legend=False)

plt.xlabel('HouseYear')

plt.annotate('1977, 25%!!', xy=(1979, 0.045)) #for picture

plt.suptitle('Distribution of HouseYear')

plt.show()
X77 = X[X['HouseYear'] == 1977] 




var = 'Rooms'

data = pd.concat([X77['Price'], X[var]], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x=var, y="Price", data=data)

fig.axis(ymin=0, ymax=600000);

plt.xticks(rotation=90);
X77.describe()
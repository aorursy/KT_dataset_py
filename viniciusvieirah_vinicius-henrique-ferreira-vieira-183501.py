import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



black_friday = pd.read_csv('../input/BlackFriday.csv', delimiter=',')

black_friday.dtypes
black_friday.drop(['Product_Category_1', 'Product_Category_2', 'Product_Category_3'], axis = 1, inplace = True)

black_friday



plt.figure(figsize=(10, 6))

sns.set_style('whitegrid')

sns.violinplot(x='Age', y='Purchase', cut=0, scale="count", data=black_friday.sort_values(by=['Age']))
pd.isnull(black_friday).sum()

plt.figure(figsize=(10, 6))

sns.set_style('whitegrid')

sns.violinplot(x='Age', y='Purchase', cut=0, scale="count", data=black_friday.sort_values(by=['Age']))

#arrumar a validacao para N>8 

pd.DataFrame(black_friday["Product_ID"].value_counts())

plt.figure(figsize=(10, 6))

black_friday["Product_ID"].value_counts().head(10).plot(kind='bar', title='Most popular products')

data_frame = black_friday['Occupation'].value_counts().head(5)

print(data_frame)

w = pd.DataFrame

for i, v in data_frame.iteritems():    

    if w.empty :        

        w = black_friday[black_friday['Occupation'] == i]

    else:

        w = w.append(black_friday[black_friday['Occupation'] == i])

        

plt.figure(figsize=(20, 10))

sns.boxenplot(x=w['Occupation'], y=w['Purchase'], hue=w['Age'], linewidth=5)
valor = black_friday[black_friday['Purchase'] > 9000]

sns.catplot(x='Marital_Status', y='Purchase', hue='Marital_Status',  margin_titles=True,

            kind='violin', col='Occupation', data=valor, aspect=.4, col_wrap=7,)
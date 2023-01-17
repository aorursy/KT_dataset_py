import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/database.csv')

x = df['armed'].value_counts()

print(x)
y = df['race'].value_counts()

print (y)
# Data to plot y

labels = 'White', 'Black', 'Hispanic', 'Asian', 'Other','None'

sizes = [878, 457, 307, 24, 22, 22]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

explode = (0, 0, 0, 0, 0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, 

        startangle=140, pctdistance=1.25, labeldistance=1.4)

plt.axis('equal')

plt.show()
z = df['flee'].value_counts()

print (z)
a = df['body_camera'].value_counts()

print (a)
b = df['state'].value_counts()

df['state'].sort_values (ascending=True)

print (b)
df.groupby([df.state]).count().plot(kind='barh', figsize=(8, 6),legend=None, fontsize=8,)
c = df['signs_of_mental_illness'].value_counts()

print (c)
# Data to plot c

labels = 'No Sign of Mental Illness', 'Signs of Mental Illness',

sizes = [1395, 456,]

colors = ['purple', 'yellow',]

explode = (0, 0,)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140,)

plt.axis('equal')

plt.show()
df.loc[df.signs_of_mental_illness==True].armed.value_counts()
df.loc[df.signs_of_mental_illness==True].flee.value_counts()
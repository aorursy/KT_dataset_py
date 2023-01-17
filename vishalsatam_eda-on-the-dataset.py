# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/car_ad.csv',encoding='latin-1')

df.head()
import matplotlib.pyplot as plt

#df.groupby('engType').engType.len(nunique())

ct = pd.value_counts(df['engType'].values, sort=False)

#print(ct)



labels = df['engType'].unique()

labels.sort()

sizes = ct

ct.sort_index(inplace=True)

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

#explode = (0.1, 0, 0, 0)  # explode 1st slice 

# Plot

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

 

plt.axis('equal')

plt.show()

import matplotlib.pyplot as pltd

y_pos = np.arange(len(labels))

pltd.bar(y_pos,sizes)

pltd.xticks(y_pos,labels)

pltd.show()
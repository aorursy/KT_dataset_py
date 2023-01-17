# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_2015 = pd.read_csv('../input/2015.csv')

df_2016 = pd.read_csv('../input/2016.csv')
df_2015.head()
df_2016.head()
f = df_2016.iloc[:30].plot(x='Country', y='Happiness Score', kind='bar', legend=False)

f.set_ylabel('Happiness Score')

f.set_title('2016 top 30 ranked happy countries')

plt.show()



cols = df_2016.columns.tolist()

removes = ['Country', 'Region', 'Happiness Rank', 'Happiness Score', 'Lower Confidence Interval',

            'Upper Confidence Interval']

nothing = [cols.remove(c) for c in removes]

for c in cols:

    f = df_2016.iloc[:30].plot(x='Country', y=c, kind='bar', legend=False)

    f.set_ylabel(c)

    plt.show()
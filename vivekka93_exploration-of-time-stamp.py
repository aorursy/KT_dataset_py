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
with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

%pylab inline

pylab.rcParams['figure.figsize'] = (12, 8)
df.head(2)
df['id'].nunique()
df.id.value_counts().hist(bins = 100)
plt.hist(df['y'], bins = 100)

plt.xlabel('Target attribute (y)')

plt.ylabel('Count')
plot3 = sns.barplot(x = df[df['id'] == 10]['timestamp'], y = df[df['id'] == 10]['y'])

plot3.set(xlabel = 'Timestamp', ylabel = 'Y value')

for tick in plot3.get_xticklabels():

    tick.set_rotation(90)
plot4 = sns.barplot(x = df[df['id'] == 306]['timestamp'], y = df[df['id'] == 306]['y'])

plot4.set(xlabel = 'Timestamp', ylabel = 'Y value')

for tick in plot3.get_xticklabels():

    tick.set_rotation(90)
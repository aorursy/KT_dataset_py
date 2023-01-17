# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df =pd.read_csv('../input/train.csv')

df.shape
np.log10(df['goal']).plot.hist(bins = 35)

plt.title('Kickstarter Goals are on a Logarithmic Scale')

plt.xlabel('log10 Goal')
df['loggoal'] = np.log10(df['goal'])

sns.lmplot(x = 'loggoal', y = 'backers_count', col ='final_status', data = df, fit_reg = False)

sns.lmplot(x = 'loggoal', y = 'backers_count', data = df[df['final_status'] == 0], fit_reg = False)

plt.title('Unsuccesful Kickstarters rarely find backers with goals above $1 Million')
df[df['goal']>1000000].shape # in fact only 232 projects had goals over $1million
df[df['goal']>1000000]['final_status'].value_counts() # and only 4 of those projects got funded
plt.figure(figsize = (6,6))

sns.boxplot(x ='final_status', y = 'loggoal', data = df)

plt.title('Successful Kickstarters have on average lower Goals')
df.groupby('final_status')['goal'].median().T
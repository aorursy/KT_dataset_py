# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/cereal.csv")

df.head()
kellogs_calories = df.loc[df['mfr'] == 'K']['calories']

post_calories = df.loc[df['mfr'] == 'P']['calories']



kellogs_rating = df.loc[df['mfr'] == 'K']['rating']

post_rating = df.loc[df['mfr'] == 'P']['rating']
sns.distplot(kellogs_calories, kde = False)
sns.distplot(post_calories, kde = False)
sns.distplot(kellogs_rating, kde = False)
sns.distplot(post_rating, kde = False)
print("calories for kellogs vs post: ")

print(ttest_ind(kellogs_calories, post_calories, equal_var=False))

print("")

print("rating for kellogs vs post: ")

print(ttest_ind(kellogs_rating, post_rating, equal_var=False))
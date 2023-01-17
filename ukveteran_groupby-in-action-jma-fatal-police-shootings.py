# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import itertools
plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv',encoding='cp1252')
df.head()
df.groupby(['armed'])
df
plt.clf()
df.groupby('manner_of_death').size().plot(kind='bar')
plt.show()
plt.clf()
df.groupby('threat_level').size().plot(kind='bar')
plt.show()
plt.clf()
df.groupby('signs_of_mental_illness').size().plot(kind='bar')
plt.show()
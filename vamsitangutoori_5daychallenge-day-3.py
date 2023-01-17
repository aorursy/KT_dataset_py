# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import scipy.stats as st

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/cereal.csv')

print(df.info())

print('Ttest values')

print(st.ttest_ind(df['sodium'],df['carbo'],equal_var=False))



plt.figure(figsize=(12,10))

plt.subplot(211)

plt.title('Sodium histogram')

plt.xlabel('Sodium')

sns.distplot(df['sodium'],hist=True,kde=False,bins = 25)

plt.subplot(212)

plt.title('Carbo histogram')

plt.xlabel('Carbo')

sns.distplot(df['carbo'],hist=True,kde=False,bins = 25)

# Any results you write to the current directory are saved as output.
df.head()

#df['carbo'].std()



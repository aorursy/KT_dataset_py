# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np   # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename),"t1")

# Any results you write to the current directory are saved as output.
df = pd.read_csv(os.path.join(dirname, filename))

print(df)
x = df['YearsExperience']

y = df['Salary']
x.shape
y.shape
x_matrix = x.values.reshape(-1,1)

x_matrix.shape
reg = LinearRegression()
reg.fit(x_matrix,y)
reg.score(x_matrix,y)
reg.coef_
reg.intercept_
reg.predict([[3]])
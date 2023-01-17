import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import seaborn as sns

sns.set(style="whitegrid")

from collections import Counter

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings('ignore')



dat = pd.read_csv('../input/uk-bank-customers/P6-UK-Bank-Customers.csv')
dat
import seaborn as sns; sns.set(color_codes=True)

ax = sns.regplot(x="Age", y="Balance", data=dat)
ax = sns.scatterplot(x="Age", y="Balance",hue="Gender", data=dat)
ax = sns.regplot(x="Age", y="Balance", x_estimator=np.mean, data=dat)
ax = sns.regplot(x="Age", y="Balance",x_bins=4, data=dat)
ax = sns.regplot(x="Age", y="Balance",x_estimator=np.mean, logx=True, data=dat)
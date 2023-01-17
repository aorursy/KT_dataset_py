import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



dat = pd.read_csv('../input/cost-function-for-electricity-producers/Electricity.csv')

dat.info()

dat.head()
dat.columns
dat.plot(kind='scatter', x='cost', y='q',alpha = 0.5,color = 'red')

plt.xlabel('Cost', fontsize=16)              # label = name of label

plt.ylabel('q', fontsize=16)

plt.title('Cost vs q Scatter Plot', fontsize=20) 
dat.cost.plot(kind = 'hist',bins = 10,figsize = (15,15))
sns.distplot(dat.cost, kde=True, label='Cost', hist_kws={"histtype": "step", "linewidth": 3,

                  "alpha": 1, "color": sns.xkcd_rgb["azure"]})



plt.title('Cost', fontsize=20)      

plt.show()
sns.distplot(dat.q, kde=True, label='q', hist_kws={"histtype": "step", "linewidth": 3,

                  "alpha": 1, "color": sns.xkcd_rgb["azure"]})



plt.title('q', fontsize=20)      

plt.show()
cont_col= ['q','pl','sl','pk','sk']

sns.pairplot(dat[cont_col], kind="reg", diag_kind = "kde")

plt.show()
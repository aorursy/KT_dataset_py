import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt



shoes = pd.read_csv("../input/7210_1.csv")
shoes.info()
shoes.describe(include=[np.number])

#shoes.describe(exclude=[np.number])
shoes.isnull().sum()
shoes.colors.value_counts().head()
shoes["midprices"]=(shoes["prices.amountMax"]+shoes["prices.amountMin"])/2

shoes["midprices"].describe()
test = shoes[["id","brand","categories","midprices","colors"]]

test.head()
pink= shoes[shoes.colors =="Pink"]

notpink=shoes[shoes.colors!="Pink"]
ttest_ind(pink.midprices,notpink.midprices,equal_var=False)
pink["midprices"].describe()
pink.midprices.plot.hist(bins=100)

plt.xlim(0,500)
notpink["midprices"].describe()
notpink.midprices.plot.hist(bins = 1000)

plt.xlim(0,500)
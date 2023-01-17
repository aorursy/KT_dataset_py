%matplotlib inline



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



def chunks(l, n):

    """Yield successive n-sized chunks from l."""

    for i in range(0, len(l), n):

        yield l[i:i + n]



df_train = pd.read_csv("../input/train.csv")

y = list(df_train["is_duplicate"])



means = []

for chunk in chunks(y, 10000):

    means.append(np.mean(chunk))



plt.plot(means)
means = []

for chunk in chunks(y, 750):

    means.append(np.mean(chunk))



plt.plot(means)
print(np.mean(y))

print(np.mean(np.r_[y, np.zeros(500000)]))
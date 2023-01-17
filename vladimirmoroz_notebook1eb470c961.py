import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
data = pd.read_csv('../input/7210_1.csv',low_memory=False)

df = pd.DataFrame.describe(data)
col = df['prices.amountMax']

hist = plt.hist(col,bins = 15)

plt.title('Distribution of shoes amount')
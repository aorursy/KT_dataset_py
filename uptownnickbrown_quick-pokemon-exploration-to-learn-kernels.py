import matplotlib

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from matplotlib import cm as cm

%matplotlib inline
# in Kaggle Kernels, input files are available at '../input/{filename}

f = '../input/Pokemon.csv'
df = pd.read_csv(f)

# df.describe
subset=df[0:10]

subset
df.drop('#', axis=1, inplace=True)

df.drop('Type 1', axis=1, inplace=True)

df.drop('Type 2', axis=1, inplace=True)

df.drop('Total', axis=1, inplace=True)

df.drop('Generation', axis=1, inplace=True)

df.drop('Legendary', axis=1, inplace=True)

subset=df[0:10]

subset
df.corr()
fig = plt.figure(figsize=(18, 16))

ax1 = fig.add_subplot(111)

cmap = cm.get_cmap('bwr', 30)

cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)

ax1.grid(False)

plt.title('Pokemon Feature Correlation')

labels=['Name','HP','Attack','Defense','Sp. Att.','Sp. Def.','Speed']

ax1.set_xticklabels(labels,fontsize=14)

ax1.set_yticklabels(labels,fontsize=14)

# Add colorbar, make sure to specify tick locations to match desired ticklabels

fig.colorbar(cax, ticks=[.25,.5,.75,.8,.85,.90,.95,1])

plt.show()
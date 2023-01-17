import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn; seaborn.set()

%matplotlib inline
data = pd.read_csv("../input/president_heights.csv")

data.head()
heights = np.array(data['height(cm)'])

print(heights)
print("Mean height       : ",heights.mean())

print("Standard deviation: ",heights.std())

print("Mininum height    : ",heights.min())

print("Maxinum height    : ",heights.max())
print("25th percentile: ",np.percentile(heights,25))

print("Median         : ",np.percentile(heights,50))

print("75th percentile: ",np.percentile(heights,75))
plt.hist(heights)

plt.title('Height Distribution of US Presidents')

plt.xlabel('height (cm)')

plt.ylabel('number');
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Simulate the rolling of dice



rolls = np.random.choice(a=[1,2,3,4,5,6], size=1000, replace=True, p=[1/6,1/6,1/6,1/6,1/6,1/6])

rolls.mean()
# Roll the dice 1 to 1000 times, The first simulation, you roll the dice once;  the second simulation, you roll the dice twice; ... the 1000th simulation, you roll the dice 1000 times.

# Each time, calculate the mean and append the means to a list variable



means = []

for i in range(1,1000):

    rolls = np.random.choice(a=[1,2,3,4,5,6], size=i, replace=True, p=[1/6,1/6,1/6,1/6,1/6,1/6])

    means.append(rolls.mean())

# Make a plot that shows how the means fluctuate/converge as the number of rolls increase

fig, ax = plt.subplots(figsize=(12,8))

ax.set_ylim((1,6))

ax.plot(means)
%matplotlib inline
# Load in our libraries

import pandas as pd # pandas for dataframes

from scipy.stats import ttest_ind # just the t-test from scipy.states

from scipy.stats import probplot # for a qq-plot

import matplotlib.pyplot as plt # plotting

import pylab



# read in our data

cereals = pd.read_csv("../input/cereal.csv")

# check out the first few lines

cereals.head()
# plot a qqplot to check normality -- most points should be along the 

# center diagonal -- if data Normally distrib

probplot(cereals["sodium"], dist="norm", plot=pylab)
cereals["sodium"][cereals["type"] == "C"][:10]
# get sodium for hot cerals

hotCereals = cereals["sodium"][cereals["type"] == "H"] # boolean indexing into cereals["sodium"]

# get sodium for cold cereals

coldCereals = cereals["sodium"][cereals["type"] == "C"]



# compare them

ttest_ind(hotCereals, coldCereals, equal_var=False)
# Lets look at the Means of each group to see which is larger:

print(f'Mean Sodium for the Hot Cereals:\n{hotCereals.mean()}')

print(f'Mean Sodium for the Cold Cereals:\n{coldCereals.mean()}')
# plot cold cereals

plt.hist(coldCereals, alpha=0.5, label='cold')

# plot hot cereals

plt.hist(hotCereals, alpha=0.5, label='hot')

# add a legend

plt.legend(loc='upper right')

# add a title

plt.title("Sodium(mg) content of cereals by type")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import ttest_ind # just the t-test from scipy.stats
from scipy.stats import probplot 
import matplotlib.pyplot as plt
import pylab

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/cereal.csv")
data.head() #see what's inside to perform a t-test on
from scipy.stats import ttest_ind
probplot(data["sodium"], dist="norm", plot=pylab) #need to double check it is normally distributed before performing a t-test
# get the sodium for hot cerals
hot = data["sodium"][data["type"] == "H"]
# get the sodium for cold ceareals
cold = data["sodium"][data["type"] == "C"]

# compare them
ttest_ind(hot, cold, equal_var=False)

#mean
print("Mean of sodium content for hot cereals:")
print(hot.mean())
print("Mean of sodium content for cold cereals:")
print(cold.mean())
# plot the cold cereals
plt.hist(cold, alpha=0.5, label='cold')
# and the hot cereals
plt.hist(hot, label='hot')
# and add a legend
plt.legend(loc='upper right')
# add a title
plt.title("Sodium(mg) content of cereals by type")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import ttest_ind
from scipy.stats import probplot
import matplotlib.pyplot as plt
import pylab
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))
cereal = pd.read_csv("../input/cereal.csv")
# print(cereal)
# print(cereal.describe())
cal = cereal['calories']
name = cereal['name']

# probplot(cal, dist="norm", plot=pylab)
# Any results you write to the current directory are saved as output.

KCereals = cal[cereal["mfr"] == "K"]
GCereals = cal[cereal["mfr"] == "G"]
ttest_ind(KCereals, GCereals, equal_var=False)



print("Mean calories for the Cereals from manufacturer K")
print(KCereals.mean())
print("Mean calories for the Cereals from manufacturer G")
print(GCereals.mean())

plt.hist(GCereals, label="Cereals from G")
plt.hist(KCereals, label="Cereals from K")
plt.legend(loc="upper right")
plt.title("Calories count from cereals by manufacturer")
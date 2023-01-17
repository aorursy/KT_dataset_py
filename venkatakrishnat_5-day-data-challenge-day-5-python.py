# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from scipy import stats as sts

hab = pd.read_csv("../input/Health_AnimalBites.csv")
# print(hab)
gender = hab["GenderIDDesc"].value_counts()
species  = hab["SpeciesIDDesc"].value_counts()
color = hab["color"].value_counts()
sts.chisquare(gender)
#sts.chisquare(species)
#sts.chisquare(color)
contingencyTable = pd.crosstab(hab["GenderIDDesc"], hab["color"])
sts.chi2_contingency(contingencyTable)

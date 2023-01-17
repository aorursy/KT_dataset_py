# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats # statistics
import pandas as pd # dataframe

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/anonymous-survey-responses.csv")
data.head()
# one-way chi-squared test for stats background
scipy.stats.chisquare(data["Have you ever taken a course in statistics?"].value_counts())

#one-way chi-squared test for programming background
scipy.stats.chisquare(data["Do you have any previous experience with programming?"].value_counts())

# now a two-way chi-square test to test for a relationship between programming background and stats background
contingencyTable = pd.crosstab(data["Do you have any previous experience with programming?"],
                              data["Have you ever taken a course in statistics?"])
scipy.stats.chi2_contingency(contingencyTable)
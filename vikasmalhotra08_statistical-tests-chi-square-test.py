# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import chisquare, chi2_contingency, chi2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
aircraft = pd.read_csv("../input/database.csv")

aircraft.head()
# One-way chisquare test to see if states are evenly distributed

chisquare(aircraft['State'].value_counts())
# One-way chisquare test to see if states are evenly distributed

chisquare(aircraft['Species Name'].value_counts())
# Combine the engine shutdown and engine shut down entries in flight impact column

aircraft.loc[aircraft["Flight Impact"] == "ENGINE SHUT DOWN", "Flight Impact"] = "ENGINE SHUTDOWN"
# Two-way Chisquare test for the relationship between visibility and flight impact

contingencyTable = pd.crosstab(aircraft['Visibility'],aircraft['Flight Impact'])

print(contingencyTable)

stat, p, dof, expected = chi2_contingency(contingencyTable)

stat, p, dof, expected
# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
# Two-way chisquare test to see whether mourning doves and gulls cause different impact

subset = aircraft.loc[aircraft['Species Name'].isin(['MOURNING DOVE','GULL'])]



bird_impact = pd.crosstab(subset["Species Name"], subset["Flight Impact"])

stat, p, dof, expected = chi2_contingency(bird_impact)

stat, p, dof, expected
# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
# Two-way Chisquare test for the relationship between Aircraft Make and flight impact

contingencyTable = pd.crosstab(aircraft['Aircraft Make'],aircraft['Flight Impact'])

print(contingencyTable)

stat, p, dof, expected = chi2_contingency(contingencyTable)

stat, p, dof, expected
# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
# Two-way Chisquare test for the relationship between Operator and flight impact

contingencyTable = pd.crosstab(aircraft['Operator'],aircraft['Flight Impact'])

print(contingencyTable)

stat, p, dof, expected = chi2_contingency(contingencyTable)

stat, p, dof, expected
# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
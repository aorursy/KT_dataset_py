# First, we'll import pandas, a data processing and CSV file I/O library

import pandas as pd



# We'll also import seaborn, a Python graphing library

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)



# Next, we'll load the Iris flower dataset, which is in the "../input/" directory

noshow = pd.read_csv("../input/No-show-Issue-Comma-300k.csv") # the iris dataset is now a Pandas DataFrame



# Let's see what's in the iris data - Jupyter notebooks print the result of the last thing you do

noshow.head()



# Press shift+enter to execute this cell

# Let's see how many values we have for the Status (dependent variable)

noshow["Status"].value_counts()
# Let's have a look at the data and find any data related issues

summary = noshow.describe()

summary = summary.transpose()

summary.head()
# Find out how many outliers we are dealing with here

outliers = noshow[(noshow['Age'] < 0) | (noshow['Age'] > 100)]

summary = outliers.describe()

summary = summary.transpose()

summary.head()

# Remove 33 records which are either bad data or outliers

# This should help with getting prediction model as we are keeping most "common" data





summary = noshow.describe()

summary = summary.transpose()

summary.head()
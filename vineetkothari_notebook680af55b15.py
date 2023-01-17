# First, we'll import pandas, a data processing and CSV file I/O library

import pandas as pd



# We'll also import seaborn, a Python graphing library

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)



# Next, we'll load the Iris flower dataset, which is in the "../input/" directory

food = pd.read_csv("../input/FoodFacts.csv") # the iris dataset is now a Pandas DataFrame



# Let's see what's in the iris data - Jupyter notebooks print the result of the last thing you do

food.head()



# Press shift+enter to execute this cell
# Let's see how many examples we have of each species

food["creator"].value_counts()

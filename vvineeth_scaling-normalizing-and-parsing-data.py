import numpy as np

import pandas as pd





# for Box-Cox Transformation

from scipy import stats



# for min_max scaling

from mlxtend.preprocessing import minmax_scaling



# plotting modules



import matplotlib.pyplot as plt

import seaborn as sns



kickstarters = pd.read_csv('../input/kickstarter2018nlp/ks-projects-201801-extra.csv')



# set seed for reproducibility

np.random.seed(0)
# generate 1000 data points randomly drawn from an exponential distribution

original_data = np.random.exponential(size=1000)



# mix-max scale the data between 0 and 1

scaled_data = minmax_scaling(original_data, columns=[0])





# plot both together to compare

fig, ax= plt.subplots(1,2)

sns.distplot(original_data, ax=ax[0])

ax[0].set_title("Original_data")

sns.distplot(scaled_data, ax=ax[1])

ax[1].set_title("Scaled_data")
# normalize the exponential data with boxcox

normalized_data = stats.boxcox(original_data)



# plot both together to compare

fig, ax = plt.subplots(1,2)

sns.distplot(original_data, ax=ax[0])

ax[0].set_title("Original_data")

sns.distplot(normalized_data[0], ax=ax[1])

ax[1].set_title("Normalized_data")
# select the usd_goal_real column

usd_goal = kickstarters.usd_goal_real



# scale the goals from 0 to 1

scaled_data = minmax_scaling(usd_goal, columns=[0])





# plot the original & scaled data together to compare

fig, ax = plt.subplots(1,2)

sns.distplot(usd_goal, ax=ax[0])

ax[0].set_title("Original_data")

sns.distplot(scaled_data, ax=ax[1])

ax[1].set_title("Scaled_data")
# get the index of all positive pledges (Box-Cox only takes postive values)

index_of_positive_pledges = kickstarters.usd_pledged_real > 0



# get only positive pledges (using their indexes)

positive_pledges = kickstarters.usd_pledged_real.loc[index_of_positive_pledges]



# normalize the pledges (w/ Box-Cox)

normalized_pledges = stats.boxcox(positive_pledges)[0]



# plot both together to compare

fig, ax=plt.subplots(1,2)

sns.distplot(positive_pledges, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(normalized_pledges, ax=ax[1])

ax[1].set_title("Normalized data")
import datetime



# Read in our dataset

landslides = pd.read_csv("../input/landslide-events/catalog.csv")



# set seed for reproducibility

np.random.seed(0)
# print the first few rows of the date column

print(landslides['date'].head())
# check the data type of our date column

landslides['date'].dtype
# create a new column, date_parsed, with the parsed dates

landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows

landslides['date_parsed'].head()
# get the day of the month from the date_parsed column

day_of_month_landslides = landslides['date_parsed'].dt.day
# remove na's

day_of_month_landslides = day_of_month_landslides.dropna()
# plot the day of the month

sns.distplot(day_of_month_landslides, kde = False, bins=30)
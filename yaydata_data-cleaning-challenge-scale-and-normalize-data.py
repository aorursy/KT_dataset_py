# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
kickstarters_2017 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

# set seed for reproducibility
np.random.seed(0)
# What are the dimensions of the data frame?
kickstarters_2017.shape
# Let's look at the data.
kickstarters_2017.head()
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)

# min-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")
# select the usd_goal_real column
usd_goal = kickstarters_2017.usd_goal_real

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
# Your turn! 

# We just scaled the "usd_goal_real" column. What about the "goal" column?
goals = kickstarters_2017.goal
# I'm not sure why the line below doesn't work.
#scaled_goals = minmax_scaling(kickstarters_2017, columns = ['goal'])
scaled_goals = minmax_scaling(goals, columns=[0])

fig, ax = plt.subplots(1,2)
sns.distplot(goals, ax=ax[0])
ax[0].set_title("Original Goals")
sns.distplot(scaled_goals, ax=ax[1])
ax[1].set_title("Scaled Goals")
# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")
# Your turn! 
# We looked as the usd_pledged_real column. What about the "pledged" column? Does it have the same info?
pledged = kickstarters_2017.pledged.where(kickstarters_2017['pledged'] > 0).dropna()
bc_normalized_pledged = stats.boxcox(pledged)[0] #if the [0] isn't included at the end of this line the variable wont be returned as a data frame
# lets also compare a natural log normalization
ln_normalized_pledged = np.log(pledged)
# Lets also compare a standard normal transformation (x-mean)/stdev
z_normalized_pledged = (pledged-pledged.mean())/np.std(pledged)

# Plot the 4 datasets to compare the normalization techniques.
fig, ax=plt.subplots(1,4, figsize=(20,7))
sns.distplot(pledged, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(bc_normalized_pledged, ax=ax[1])
ax[1].set_title("Box-Cox Normalized Data")
sns.distplot(ln_normalized_pledged, ax=ax[2])
ax[2].set_title("Natural Log Normalized Data")
sns.distplot(z_normalized_pledged, ax=ax[3])
ax[3].set_title("Z Normalized Data")
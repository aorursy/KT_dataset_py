import matplotlib.pyplot as plt

import numpy as np

import pandas as pd 

import seaborn as sns
tips = sns.load_dataset("tips") # get the tips dataset from the internet

tips # A small dataset compared to our Lending Club dataset :)
# Here we will plot tip against total_bill 

sns.relplot(x="total_bill", y="tip", data=tips); # Note here x="" and y="" corresponds to the column names in the tips dataset 
# Method 1. 

fig, ax = plt.subplots()

sns.scatterplot(x="total_bill", y="tip", data=tips)

ax.set_xlabel("Total Bill")

ax.set_ylabel("Tip")

ax.set_title("Tip vs Total Bill");
#Method 2. 

g = sns.relplot(x="total_bill", y="tip", data=tips)

g.set_axis_labels("Total Bill", "Tip")

g.ax.set_title("Tip vs Total Bill");
sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);
# Categorical scatterplots

# Here we show how to plot total_bill within each day 

# To prevent from overlapping try kind="swarm"

sns.catplot(x="day", y="total_bill", data=tips);
# Categorical distribution plots (distributions of observations within categories)

sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips);
# Categorical estimate plots

# Here the height of the bars represent the mean (by default) and the error bars represent a confidence interval around the estimate

sns.catplot(x="day", y="tip", hue="sex", kind="bar", data=tips);
# Here we examine the univariate distribution of tip using histograms

sns.distplot(tips["tip"], kde=False);
# Here we add a kernel density estimate (KDE) on top of the histogram.

sns.distplot(tips["tip"]);
# Load in the data 

train = pd.read_csv("../input/ucfai-dsg-fa19-default/train.csv") #we could set low_memory=True but it's useful to see that column numbers for exploration

test = pd.read_csv("../input/ucfai-dsg-fa19-default/test.csv")
# Take a look at the data with describe 

# YOUR CODE HERE

raise NotImplementedError()
# Take a look at the categorical data

# YOUR CODE HERE

raise NotImplementedError()
# Count number of columns with at least one missing value

# YOUR CODE HERE

raise NotImplementedError()
# Percent columns that contain at least one missing value

# YOUR CODE HERE

raise NotImplementedError()
# Count number of row with at least one missing value

# YOUR CODE HERE

raise NotImplementedError()
# Calculate the percentage of rows that contain at least one missing value

# YOUR CODE HERE

raise NotImplementedError()
# Count number of columns with at least one missing value

# YOUR CODE HERE

raise NotImplementedError()
# Calculate the percentage of nulls in each column 

# YOUR CODE HERE

raise NotImplementedError()
# Find the number of columns that have over 90% nulls in them 

# YOUR CODE HERE

raise NotImplementedError()
#visualize loan_amnt, funded_amnt, and funded_amnt_inv for the train dataset 

# YOUR CODE HERE

raise NotImplementedError()
#visualize loan_amnt, funded_amnt, and funded_amnt_inv for the test dataset 

# YOUR CODE HERE

raise NotImplementedError()
# Plot the number of loans issued per month

# YOUR CODE HERE

raise NotImplementedError()
# Convert "issue_d" to pandas datetime format for the train set and then plot the number of loans issued per year 

# YOUR CODE HERE

raise NotImplementedError()
# Convert "issue_d" to pandas datetime format for the test set and then plot the number of loans issued per year 

# YOUR CODE HERE

raise NotImplementedError()
# Plot the number of loans within each grade category for the train set

# YOUR CODE HERE

raise NotImplementedError()
# Plot the number of loans within each grade category for the test set

# YOUR CODE HERE

raise NotImplementedError()
# Plot the number of loans within each grade category with hue train. 

# YOUR CODE HERE

raise NotImplementedError()
# Plot the number of loans within each subgrade with hue GOOD_STANDING

# YOUR CODE HERE

raise NotImplementedError()
# import pandas_profiling



# train.profile_report()
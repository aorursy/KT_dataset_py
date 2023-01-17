import pandas as pd
import matplotlib.pyplot as plt

# importing the data
cars_data=pd.read_csv('../input/aqalds/AQA-large-data-set.csv')

# inspecting the dataset to check that it has imported correctly
cars_data.head()
# check the datatypes

# use describe for any fields you are going to investigate and filter out any unusable values

# replace body type and propulsion type with appropriate text values
# find the means and standard deviations for different groups

# create box plots for the different groups
# communicate the result
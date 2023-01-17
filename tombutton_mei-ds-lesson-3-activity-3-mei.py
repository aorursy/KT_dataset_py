import pandas as pd
import matplotlib.pyplot as plt

# importing the data
health_data=pd.read_csv('../input/meilds3/mei-lds-3.csv')

# inspecting the dataset to check that it has imported correctly
health_data.head()
# check the datatypes

# use describe for any fields you are going to investigate and filter out or replace any unusable values
# find the means and standard deviations for different fields grouped by sex or marital status

# create box plots for different fields grouped by sex or marital status
# communicate the result
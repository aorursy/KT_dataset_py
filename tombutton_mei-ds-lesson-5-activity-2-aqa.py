import pandas as pd
import matplotlib.pyplot as plt

# importing the data
cars_data=pd.read_csv('../input/aqalds/AQA-large-data-set.csv')

# inspecting the dataset to check that it has imported correctly
cars_data.head()
# check the datatypes

# use describe() and/or boxplots for any fields you are going to investigate and filter out any unusable values

# create scatter diagrams for CO2 v CO, CO2 v NOX and CO v NOX

# create scatter diagrams for two of the emissions on the x and y axis and the third as size or colour

# create scatter diagrams for two of the emissions with a colour map for the make
# Communicate the result
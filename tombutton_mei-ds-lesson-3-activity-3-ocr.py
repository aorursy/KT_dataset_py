import pandas as pd
import matplotlib.pyplot as plt

# importing the data
travel_2011_data=pd.read_csv('../input/ocrlds/OCR-lds-travel-2011.csv')

# inspecting the dataset to check that it has imported correctly
travel_2011_data.head()
# check the datatypes

# use describe for any fields you are going to investigate and filter out or replace any unusable values
# find the means and standard deviations for different fields grouped by region

# create box plots for different fields grouped by region
# communicate the result
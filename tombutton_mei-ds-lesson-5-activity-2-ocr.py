import pandas as pd
import matplotlib.pyplot as plt

# importing the data
travel_2011_data=pd.read_csv('../input/ocrlds/OCR-lds-travel-2011.csv')

# inspecting the dataset to check that it has imported correctly
travel_2011_data.head()
# check the datatypes

# use describe and/or boxplots for any fields you are going to investigate and filter out or replace any unusable values (e.g. removing commas and converting to floating point numbers)
# create scatter diagrams for various fields plotted against cycling

# create scatter diagrams for various fields plotted against cycling with a further field controlling the size or colour
# communicate the result
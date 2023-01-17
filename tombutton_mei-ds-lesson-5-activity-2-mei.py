import pandas as pd
import matplotlib.pyplot as plt

# importing the data
health_data=pd.read_csv('../input/meilds3/mei-lds-3.csv')

# inspecting the dataset to check that it has imported correctly
health_data.head()
# check the datatypes

# use describe and/or boxplots for any fields you are going to investigate and filter out or replace any unusable values
# create scatter diagrams for various fields plotted against systolic blood pressure

# create scatter diagrams for various fields plotted against systolic blood pressure with a further field controlling the size or colour
## communicate the result
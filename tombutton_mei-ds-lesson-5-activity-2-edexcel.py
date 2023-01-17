# import the pandas and matplotlib modules

import pandas as pd
import matplotlib.pyplot as plt

# copy data from a file called heathrow-2015.csv and store it in pandas as a dataset called heathrow_2015_data

heathrow_2015_data = pd.read_csv("../input/ldsedexcel/heathrow-2015.csv")

# show the first 6 records from the data

heathrow_2015_data.head(6)
# check the datatypes

# use describe for any fields you are going to investigate and filter out or replace any unusable values (such as 'tr')
# create scatter diagrams for various fields plotted against rainfall

# create scatter diagrams for various fields plotted against rainfall with a further field controlling the size or colour

# try importing the data for a different year or weather station and explore if the relationships are similar
# communicate the result
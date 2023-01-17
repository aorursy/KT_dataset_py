#import packages and liabraries

#importing unicodecsv to read and load the data file
import unicodecsv

#importing numpy as np
import numpy as np

#importing pandas as pd
import pandas as pd

#importing matplotlib.pyplot as plt
import matplotlib.pyplot as plt

#importing seaborn as sns
import seaborn as sns

#display code
from IPython.display import display

#matplot library holding in Ipyton Notebooks
%matplotlib inline
#import titanic_data.csv dataframe through pandas inside titanic_data
titanic_data = pd.read_csv('titanic_data.csv')

#Displaying the small dataframe through the Head function
display(titanic_data.head())

print("")

#DataFrame Information
titanic_data.info()
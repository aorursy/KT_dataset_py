import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex7 import *
print("Setup Complete")
# Check for a dataset with a CSV file
step_1.check()
# Fill in the line below: Specify the path of the CSV file to read
my_filepath = "../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv"

# Check for a valid filepath to a CSV file in a dataset
step_2.check()
# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")

# Check that a dataset has been uploaded into my_data
step_3.check()
# Print the first five rows of the data
my_data.head()
# Create a plot

plt.figure(figsize=(8, 6))

sns.barplot(x=my_data['blueWardsPlaced'], y=my_data.index)
# Add label for horizontal axis
plt.xlabel("")
# Add label for vertical axis
plt.title("Average blueWardsPlaced for games, by Platform")


# Check that a figure appears below
step_4.check()
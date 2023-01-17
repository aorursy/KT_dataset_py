import pandas as pd

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

my_filepath = "../input/heart-disease-uci/heart.csv"

# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)

# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()

my_data.describe()
# Create a plot

plt.figure(figsize=(10,2))



# Add title

plt.title("Percentage of males and females")

# chart showing the Percentage of males and females

sns.kdeplot(data=my_data['sex'], shade=True)

#sns.barplot(x=my_data.index, y=my_data['sex'])



# Add label for vertical axis

#plt.ylabel("Percentage of males and females")

# scater plot

#sns.scatterplot(x=my_data['age'], y=my_data['sex'])



# Check that a figure appears below

step_4.check()
# Histogram 

sns.distplot(a=my_data['sex'], kde=False)

plt.title("Percentage of males and females")
sns.swarmplot(x=my_data['sex'],

              y=my_data['age'])
# Histogram 

sns.distplot(a=my_data['age'], kde=False)




# 2D KDE plot

sns.jointplot(x=my_data['age'], y=my_data['chol'], kind="kde")







# 2D KDE plot

sns.jointplot(x=my_data['sex'], y=my_data['chol'], kind="kde")



sns.swarmplot(x=my_data['sex'],

              y=my_data['chol'])
sns.lmplot(x="age", y="chol", hue="exang", data=my_data)
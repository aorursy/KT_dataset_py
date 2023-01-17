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

# step_1.check()
# Fill in the line below: Specify the path of the CSV file to read

my_filepath = "../input/graduate-admissions/Admission_Predict_Ver1.1.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
# Create a plot

plt.figure(figsize=(12,8)) # Your code here

plt.title("Graduate Admissions Statistics (GPA) and Rates") 

sns.scatterplot(x=my_data["CGPA"], y=my_data["Chance of Admit "])

sns.regplot(x=my_data["CGPA"], y=my_data["Chance of Admit "])

plt.legend()



plt.figure(figsize=(12,6))

plt.title("Graduate Admissions Statistics (Research) and Rates") 

sns.swarmplot(x=my_data["Research"], y=my_data["Chance of Admit "])

plt.legend()



# Check that a figure appears below

step_4.check()
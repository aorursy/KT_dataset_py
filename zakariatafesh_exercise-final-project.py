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

my_filepath = '../input/netflix-shows/netflix_titles.csv'

my_filepath = '../input/numerical-dataset/phpB0xrNj.csv'



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data = my_data.dropna().head()

my_data
s = (my_data.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)

from sklearn.preprocessing import OneHotEncoder



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(my_data[object_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = my_data.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = my_data.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

ll = ['f1','f2','f3']

OH_X_train[ll]
# Create a plot

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



sns.set_style("dark")



# Line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=OH_X_train[ll])



# Check that a figure appears below

step_4.check()
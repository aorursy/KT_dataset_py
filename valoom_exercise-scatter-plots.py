import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

import os

if not os.path.exists("../input/candy.csv"):

    os.symlink("../input/data-for-datavis/candy.csv", "../input/candy.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex4 import *

print("Setup Complete")
# Path of the file to read

candy_filepath = "../input/candy.csv"



# Fill in the line below to read the file into a variable candy_data

candy_data = pd.read_csv(candy_filepath, index_col='id')



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the first five rows of the data

# Your code here

candy_data.head()
# Fill in the line below: Which candy was more popular with survey respondents:

# '3 Musketeers' or 'Almond Joy'?  (Please enclose your answer in single quotes.)

more_popular = '3 Musketeers'



# Fill in the line below: Which candy has higher sugar content: 'Air Heads'

# or 'Baby Ruth'? (Please enclose your answer in single quotes.)

more_sugar = 'Air Heads'



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'

# Your code here

sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])

# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
#step_3.b.hint()

# I don't think we can conclude any correlation between these variables with this result.
step_3.b.solution()
# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'

# Your code here

sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])

# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()

# It may be a slight, maybe non-relevant correlation. It would tell me that with more sugar, more winpercent.
step_4.b.solution()
# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'

# Your code here

plt.figure(figsize=(12,5))

sns.scatterplot(x=candy_data.pricepercent, y=candy_data.winpercent, hue=candy_data.chocolate)

# Check your answer

step_5.check()
# Lines below will give you a hint or solution code

#step_5.hint()

#step_5.solution_plot()
# Color-coded scatter plot w/ regression lines

# Your code here

sns.lmplot(x='sugarpercent', y='winpercent', hue='chocolate', data=candy_data)

# Check your answer

step_6.a.check()
# Lines below will give you a hint or solution code

#step_6.a.hint()

#step_6.a.solution_plot()
#step_6.b.hint()

# I think we can conclude with a bigger correlation b/w chocolate and popularity
step_6.b.solution()
# Scatter plot showing the relationship between 'chocolate' and 'winpercent'

# Your code here

sns.swarmplot(x=candy_data.chocolate, y=candy_data.winpercent)

# Check your answer

step_7.a.check()
# Lines below will give you a hint or solution code

#step_7.a.hint()

#step_7.a.solution_plot()
#step_7.b.hint()

# I will consider both, but if I have to choose just one, ignoring all real facts in a real world, I would have to pick the Step 7 plot
step_7.b.solution()
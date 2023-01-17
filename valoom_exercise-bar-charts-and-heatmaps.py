import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

import os

if not os.path.exists("../input/ign_scores.csv"):

    os.symlink("../input/data-for-datavis/ign_scores.csv", "../input/ign_scores.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex3 import *

print("Setup Complete")
# Path of the file to read

ign_filepath = "../input/ign_scores.csv"



# Fill in the line below to read the file into a variable ign_data

ign_data = pd.read_csv(ign_filepath, index_col='Platform')



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the data

# Your code here

ign_data
# Fill in the line below: What is the highest average score received by PC games,

# for any platform?

high_score = max(ign_data.loc['PC'])



# Fill in the line below: On the Playstation Vita platform, which genre has the 

# lowest average score? Please provide the name of the column, and put your answer 

# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)

minim = min(ign_data.loc['PlayStation Vita'])

b = minim == ign_data.loc['PlayStation Vita']

worst_genre = [i for indx,i in enumerate(ign_data.columns) if b[indx] == True].pop()



print(worst_genre)

# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Bar chart showing average score for racing games by platform

# Your code here

plt.figure(figsize=(20,6))

ign_data = ign_data.rename({"PlayStation Vita": "PSV", "PlayStation": "PS", "Game Boy Advance": "GBA", "PlayStation Portable": "PSP",

                        "PlayStation 2": "PS2", "PlayStation 3":"PS3", "Nintendo 3DS" : "N3DS", "Nintendo DSi": "NDSi", 

                            "Nintendo DS": "NDS", "Game Boy Color": "GBC"})

sns.barplot(x=ign_data.index, y=ign_data['Racing'])

# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
#step_3.b.hint()

# Not at all, I would try with XBO, PS4 or iPhone in that order
step_3.b.solution()
# Heatmap showing average game score by platform and genre

# Your code here

plt.figure(figsize=(12, 6))

sns.heatmap(data=ign_data, annot=True)

# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()
step_4.b.solution()
import pandas as pd

# Define custom options
pd.set_option('max_rows', 5)

# Setup
from learntools.core import binder; binder.bind(globals())
from learntools.pandas.creating_reading_and_writing import *
print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
fruits = pd.DataFrame({ "Apples" : [30 ], "Bananas" : [21]})

# Check your answer
q1.check()
fruits
#q1.hint()
#q1.solution()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
columns =[ "2017 Sales", "2018 Sales"]
fruit_sales = pd.DataFrame([{"Apples" : 35, "Bananas" : 21 },
                            { "Apples" : 41, "Bananas" : 34 }], index=columns)
# index = fruit_sales.index()
# fruit_sales.index = columns


# Check your answer
q2.check()
fruit_sales
#q2.hint()
#q2.solution()
ingredients = pd.Series(["4 cups", "1 cup", "2 large", "1 can"], index=["Flour", "Milk", "Eggs", "Spam"], name = "Dinner")

# Check your answer
q3.check()
ingredients
#q3.hint()
#q3.solution()
reviews = pd.read_csv(
    "../input/wine-reviews/winemag-data_first150k.csv", delimiter= ',', index_col="Unnamed: 0")

# Check your answer
q4.check()
reviews.head()
q4.hint()
#q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals
# Your code goes here
animals.to_csv("cows_and_goats.csv")

# Check your answer
q5.check()
#q5.hint()
#q5.solution()
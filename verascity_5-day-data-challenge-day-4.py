import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../input/database.csv")
print(list(data))
species = data["Species Name"]
print(species.value_counts())
top_five = ["MOURNING DOVE", "GULL", "KILLDEER", "AMERICAN KESTREL", "BARN SWALLOW"] #Our top known five, based on the list above. 
top_five_species = species[species.isin(top_five)] #isin() returns a Boolean array, which we check our original Series (species) against to create a new Series.
print(top_five_species.value_counts()) #The new column has the same values and counts as the top five known species in the old column.
species_count = sns.countplot(top_five_species) #A countplot() is a type of bar plot specifically for value counts.
plt.title("Top Five Known Species That Impact with Aircraft") #That's kind of a morbid title, isn't it?
plt.xticks(rotation='vertical') #This rotates our x-axis labels so they don't smush together. 

import pandas as pd
import scipy.stats #to run our chi-square test

data = pd.read_csv("../input/database.csv", low_memory=False) #To prevent the mixed data-type warning we experienced yesterday, we set low_memory to False.
print(list(data))
top_five = ["MOURNING DOVE", "GULL", "KILLDEER", "AMERICAN KESTREL", "BARN SWALLOW"] #Our top known five, based on the list above. 
viz = ["DAWN", "DAY", "DUSK", "NIGHT"]

species = data["Species Name"].loc[data["Species Name"].isin(top_five)] #isin() returns a Boolean array, which we check our original Series (species) against to create a new Series.
visibility = data["Visibility"].loc[data["Visibility"].isin(viz)] #Ditto here
ctable = pd.crosstab(species, visibility)
print(ctable)
chi2, p, dof, expected = scipy.stats.chi2_contingency(ctable, correction=False)
print("Chi-Statistic: {}; P-Value: {}; DOF: {}\nExpected values: {}".format(chi2, p, dof, expected))
ctable.plot.barh(stacked=True, colormap="Dark2") #The default colors weren't colorblind-friendly; I chose a more accessible colormap. 
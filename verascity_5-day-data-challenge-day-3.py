import numpy as np
import pandas as pd #Pandas will help us read in our dataframe and look at our data before we run the test.
from scipy.stats import ttest_ind #Ttest_ind is a function of scipy that will produce a t-test.
import matplotlib.pyplot as plt #Pyplot will help us generate histograms of our variable groups.

cereal_data = pd.read_csv("../input/cereal.csv")
print(list(cereal_data))
print(cereal_data["mfr"].value_counts())
kelloggs = cereal_data.loc[cereal_data["mfr"] == "K", "sugars"]
gm = cereal_data.loc[cereal_data["mfr"] == "G", "sugars"]
ttest_ind(kelloggs, gm, equal_var=False)
plt.subplot(121) #This tells Pyplot that our plot figure will have one row and two columns of plots and that this is the first plot.
plt.hist(kelloggs)
plt.title("Sugar Content of Kelloggs")
plt.xlabel("Sugar Content in Grams")

plt.subplot(122) #This tells Pyplot that this is the second plot. 
plt.hist(gm)
plt.title("Sugar Content of General Mills")
plt.xlabel("Sugar Content in Grams")

plt.show()
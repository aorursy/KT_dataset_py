# Importing the packages

#We used pandas for Data pre processing ,Python pandas are the one of most powerfull package

import pandas as pd
#Importing the data sets to this notebook

iris = pd.read_csv("../input/Iris.csv")
# Check what is there in data, attributes name and all

iris.head()# By default it will print only first five rows

# If we need to print the last five rows iris.tail() put in code
# Count the number of variables in the dataset.

iris['Species'].value_counts()
#Checking the info of the iris data ,data types of every attributes for further steps

iris.info()
# Here i used the seaborn facetgrid to see the distribution of different species

# First we considered for the "SepalLengthCm", "SepalWidthCm"

# By Scatter plot we don't get much information

import matplotlib.pyplot as plt

import seaborn as sns

sns.FacetGrid(iris, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
# The we want to check for "PetalLengthCm","PetalWidthCm"

sns.FacetGrid(iris, hue="Species", size=5).map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend()
# Instead of showing different we will combine both graphs

# For this python seaborn given a beautiful pairplot which is called matrix graphs or we called Scatterplot Matrices

# by this for particular PetalLengthCm and SepalLengthCm there we can tell which species by looking at graph

sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)
# Instead of showing in the bar graphs we could show in line graphs 

# The seaborn package is good for visulization , In data kind =kde has to give 

# We can use regression line abd histogram also 

sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")
# Now we are trying the reg kind 

#pairplot which shows relation between each pair of attributes or variables

sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, kind="reg")
# Now we took only two attributes to perfect group

sns.pairplot(iris.drop("Id", axis=1),hue="Species", vars=["PetalWidthCm", "SepalWidthCm"])
sns.pairplot(iris.drop("Id", axis=1),hue="Species", vars=["SepalLengthCm", "PetalLengthCm"])
# Now we checking for the SepalLengthCm as individuals

sns.boxplot(x="Species", y="SepalLengthCm", data=iris)
# Instead of doing individual we combine all and show in one frame

iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))
sp = iris.drop("Id", axis=1)

iris_long = pd.melt(sp, id_vars='Species')

sns.boxplot(x='Species', y='value', hue='variable', data=iris_long)


ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)

ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")
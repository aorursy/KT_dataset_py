import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import numpy as np

%matplotlib inline



# Import statements required for Plotly 

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools



iris = pd.read_csv('../input/Iris.csv')





iris.head(10)
iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm",)
import seaborn as sns

sns.set(style="darkgrid")

iris = sns.load_dataset("iris")



# Subset the iris dataset by species

versicolor = iris.query("species == 'versicolor'")

setosa = iris.query("species == 'setosa'")

virginica = iris.query("species == 'virginica'")





# Set up the figure

f, ax = plt.subplots(figsize=(12, 12))

ax.set_aspect("equal")



# Draw the two density plots

ax = sns.kdeplot(setosa.sepal_width, setosa.sepal_length,

                 cmap="Purples", shade=False, shade_lowest=False)

ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,

                 cmap="Blues", shade=False, shade_lowest=False)

ax = sns.kdeplot(versicolor.sepal_width, versicolor.sepal_length,

                 cmap="Oranges", shade=False, shade_lowest=False)



# Add labels to the plot

purple = sns.color_palette("Purples")[-2]

blue = sns.color_palette("Blues")[-2]

orange = sns.color_palette("Oranges")[-2]

ax.text(2.5, 8.2, "virginica", size=16, color=blue)

ax.text(3.8, 4.5, "setosa", size=16, color=purple)

ax.text(1.8, 6.6, "versicolor", size=16, color=orange)
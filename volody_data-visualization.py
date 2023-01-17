import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings     # `do not disturbe` mode

warnings.filterwarnings('ignore')

print("Setup Complete")
forest_train = pd.read_csv('../input/learn-together/train.csv')

# print a summary of the data

forest_train.describe()
# Print the first 5 rows of the data

forest_train.head()
# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Average Cover Type, by Elevation")



# Bar chart showing average Cover Type for Elevation by 100 meter

bins = 100 * np.round(forest_train['Elevation']/100)

bins = bins.astype(int)

sns.barplot(x=bins, y=forest_train['Cover_Type'])



# Add label for vertical axis

plt.ylabel("Average Cover Type, by Elevation")
# Set the width and height of the figure

plt.figure(figsize=(10,6))

# Add title

plt.title("Cover Type by Elevation")



# create table

df = forest_train[['Elevation','Cover_Type']].copy()

bins = 100 * np.round(df['Elevation']/100)

bins =  bins.astype(int)

df['Elevation'] = bins

df['Type'] = 1



# use the pivot to arrange the data in a table

heat_data = df.groupby(['Elevation','Cover_Type'],as_index = False).sum().pivot('Elevation','Cover_Type').fillna(0)

heat_data['Type'] = heat_data['Type'].astype(int)

heat_data.sort_index(ascending=False, inplace=True)



# Heatmap showing average Cover Type for each Elevation by 100 meter

sns.heatmap(data=heat_data,annot=True,fmt="d")



# Add label for x and y axis

plt.ylabel("Elevation")

plt.xlabel("Cover Type")


#sns.scatterplot(x=scatter_data['Cover_Type'], y=scatter_data['Elevation'])

sns.scatterplot(x='Cover_Type', y='Elevation', data=forest_train)
# create table for histogram

items = forest_train[['Elevation','Cover_Type']].copy()

bins = 100 * np.round(items['Elevation']/100)

bins =  bins.astype(int)

items['Elevation'] = bins



items.head()
# 2D KDE plot

sns.jointplot(x=items['Cover_Type'], y=items['Elevation'], kind="kde")
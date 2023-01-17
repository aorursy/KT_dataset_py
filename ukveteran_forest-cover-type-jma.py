# Importing Libraries

import pandas as pd 

pd.set_option("display.max_columns", None)

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier, plot_importance



# Import Warnings Filter

from warnings import simplefilter

# Ignore all Future Warnings

simplefilter(action='ignore', category=FutureWarning)

import warnings

warnings.filterwarnings('ignore')
# Loading / Creating DataFrames

df= pd.read_csv("../input/forest-cover-type-dataset/covtype.csv")

df
wilderness_1 = data[data['Wilderness_Area1'] == 1]

wilderness_2 = data[data['Wilderness_Area2'] == 1]

wilderness_3 = data[data['Wilderness_Area3'] == 1]

wilderness_4 = data[data['Wilderness_Area4'] == 1]

soil = data.iloc[:, 15:55]



print("DataFrames:     Shape:")

print("- data:         ", data.shape)

print("- wilderness_1: ", wilderness_1.shape)

print("- wilderness_2: ", wilderness_2.shape)

print("- wilderness_3: ", wilderness_3.shape)

print("- wilderness_4: ", wilderness_4.shape)

print("- soil:         ", soil.shape)
# Summary Statistics (only showing first 3 variables)

data.iloc[:, :3].describe()
# Dimensions of the Dataset

data.shape
# Plotting Cover Types

sns.countplot(x='Cover_Type', data=data)

plt.title("Frequency of Cover Types")

plt.show()
# Plotting Histograms

data.iloc[:, :11].hist(figsize = (15,10))

plt.show()
# Plotting Soil Types

soil.sum().plot(kind="bar", figsize=(15,6))

plt.title("Frequency of Each Soil Type")

plt.show()
# Generates Wilderness Subplots

fig, ax = plt.subplots(2,2, figsize=(9,9))



sns.countplot(x='Cover_Type', data=wilderness_1, ax=ax[0,0])

ax[0,0].set_title(f"Wilderness 1", size=20)

ax[0,0].set_ylabel('Count')

ax[0,0].set_xlabel('Cover Type')



sns.countplot(x='Cover_Type', data=wilderness_2, ax=ax[0,1])

ax[0,1].set_title(f"Wilderness 2", size=20)

ax[0,1].set_ylabel('Count')

ax[0,1].set_xlabel('Cover Type')



sns.countplot(x='Cover_Type', data=wilderness_3, ax=ax[1,0])

ax[1,0].set_title(f"Wilderness 3", size=20)

ax[1,0].set_ylabel('Count')

ax[1,0].set_xlabel('Cover Type')



sns.countplot(x='Cover_Type', data=wilderness_4, ax=ax[1,1])

ax[1,1].set_title(f"Wilderness 4", size=20)

ax[1,1].set_ylabel('Count')

ax[1,1].set_xlabel('Cover Type')



plt.tight_layout(pad=7, w_pad=0.5, h_pad=1.5)

plt.suptitle(f"Cover Type Counts by Wilderness Number", size=20)

plt.show()
# Generates Correlation Heatmap

corr = data.iloc[:, :10].corr()

plt.figure(figsize=(10,10))

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);

plt.title("Correlation Between Variables")

plt.show()
# Plotting Elevation & Cover Types

plt.figure(figsize=(15,7))

sns.swarmplot(x='Cover_Type', y='Elevation', data=data.sample(2000))

plt.title("Comparing Elevations of Cover Types")

plt.show()
# Plotting Means

plt.figure(figsize=(7,5))

data.groupby("Cover_Type").mean()["Elevation"].plot(kind='bar')

plt.title("Mean Elevations of Cover Types")

plt.show()
# Plotting Horizontal Distance to Roadways & Cover Types

plt.figure(figsize=(15,7))

sns.swarmplot(x='Cover_Type', y='Horizontal_Distance_To_Roadways', data=data.sample(1750))

plt.title("Comparing Distance to Roadways of Cover Types")

plt.show()
# Plotting Means

plt.figure(figsize=(7,5))

data.groupby("Cover_Type").mean()["Horizontal_Distance_To_Roadways"].plot(kind='bar')

plt.title("Mean Distances to Roadways of Cover Types")

plt.show()
# Plotting Distance Variables

columns = ['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 

           'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']

g = sns.PairGrid(data.loc[:, columns].sample(1000), palette='coolwarm')

g = g.map_diag(plt.hist)

g = g.map_upper(plt.scatter, linewidths=1, edgecolor='w', s=40)

g = g.map_lower(sns.kdeplot, cmap='coolwarm')

plt.show()
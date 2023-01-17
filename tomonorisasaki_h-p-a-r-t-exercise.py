import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

#Unix commands

import os



# import useful tools

from glob import glob

from PIL import Image

import cv2



# import data visualization

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns



from bokeh.plotting import figure

from bokeh.io import output_notebook, show, output_file

from bokeh.models import ColumnDataSource, HoverTool, Panel

from bokeh.models.widgets import Tabs



# import data augmentation

import albumentations as albu



# import math module

import math
default_path = '/kaggle/input/house-prices-advanced-regression-techniques'
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
# display the MSZoning of the training data without duplicates

print(train['MSZoning'].drop_duplicates())
# Draw a pie chart about MSZoning.

plt.pie(train['MSZoning'].value_counts(),labels=['RL', 'RM', 'C', 'FV', 'RH'],autopct="%.1f%%")

plt.title("Ratio of MSZoning")

plt.show()
# display the Street of the training data without duplicates

print(train['Street'].drop_duplicates())
# Draw a pie chart about Street.

plt.pie(train['Street'].value_counts(),labels=['Pave', 'Grvl'],autopct="%.1f%%")

plt.title("Ratio of Street")

plt.show()
# display the Alley of the training data without duplicates

print(train['Alley'].drop_duplicates())
# Draw a pie chart about Alley.

plt.pie(train['Alley'].value_counts(),labels=['Grvl', 'Pave'],autopct="%.1f%%")

plt.title("Ratio of Alley")

plt.show()
# display the LotShape of the training data without duplicates

print(train['LotShape'].drop_duplicates())
# Draw a pie chart about LotShape.

plt.pie(train['LotShape'].value_counts(),labels=['Reg', 'IR1', 'IR2', 'IR3'],autopct="%.1f%%")

plt.title("LotShape")

plt.show()
# Show the correlation between age and FVC in the training data

sns.scatterplot(data=train, x='LotArea', y='SalePrice')
#Conversion of category variables to arbitrary values

train['MSZoning'] = train['MSZoning'].map({'RL': 0, 'RM': 1, 'C': 2, 'FV': 3,'RH': 4})

train['Street'] = train['Street'].map({'Pave': 0, 'Grvl': 1})

train['Alley'] = train['Alley'].map({'Pave': 0, 'Grvl': 1})

train['LotShape'] = train['LotShape'].map({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3})
df_corr = train.corr()

print(df_corr)
corr_mat = train.corr(method='pearson')

sns.heatmap(corr_mat,

            vmin=-1.0,

            vmax=1.0,

            center=0,

            annot=True, # True:Displays values in a grid

            fmt='.1f',

            xticklabels=corr_mat.columns.values,

            yticklabels=corr_mat.columns.values

           )

plt.show()
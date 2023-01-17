# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns #for visualization

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/ashrae-global-thermal-comfort-database-ii/ashrae_db2.01.csv')

data.head() #get first 5 row by default
data.info()
data.corr()
f,ax = plt.subplots(figsize = (50,50))

sns.heatmap(data.corr(), annot = True,  cmap = 'PiYG',  linewidths =1, fmt = '.2f', ax = ax)

#annot: text inside each boxes

#linewidths: width of line between each boxes

#fmt: to set decimal part of text inside each boxes

plt.show()
#Line Graph

data.PPD.plot(kind = "line", color = "r", label = "PPD", linewidth = 1, alpha = 0.5, grid = "True", linestyle = "-")

data.PMV.plot(kind = "line", color = "g", label = "PMV", linewidth = 1, alpha = 0.5, grid = "True", linestyle = "-.")

plt.legend(loc = "upper left")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Line Plot")

plt.show()
#Scatter Graph

data.plot(kind = "scatter", x = "Air temperature (C)", y = "Operative temperature (C)", alpha = 0.5, color = "red")

plt.xlabel("Air temperature (C)")

plt.ylabel("Operative temperature (C")

plt.title("Air temperature (C) Operative temperature (C Scatter Plot")

plt.show()
#Histogram Graph

#bins = number of bar in figure

data.PPD.plot(kind= "hist", bins = 50 )

plt.title("Histogram")

plt.show()
#plt.clf() for cleaning
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import warnings; warnings.filterwarnings(action='once')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



large = 22; med = 16; small = 12

params = {'axes.titlesize': large,

          'legend.fontsize': med,

          'figure.figsize': (16, 10),

          'axes.labelsize': med,

          'axes.titlesize': med,

          'xtick.labelsize': med,

          'ytick.labelsize': med,

          'figure.titlesize': large}

plt.rcParams.update(params)

plt.style.use('seaborn-whitegrid')

sns.set_style("white")

%matplotlib inline



# Version

print(mpl.__version__)  #> 3.0.0

print(sns.__version__)  #> 0.9.0
economics = pd.read_csv("../input/matplotlibmldatasets/economics.csv")

midwest = pd.read_csv("../input/matplotlibmldatasets/midwest_filter.csv")

mpg_ggplot2 = pd.read_csv("../input/matplotlibmldatasets/mpg_ggplot2.csv")
# Prepare Data 

# Create as many colors as there are unique midwest['category']

categories = np.unique(midwest['category'])

colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]
# Draw Plot for Each Category

from matplotlib.pyplot import figure

plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')

for i, category in enumerate(categories):

    plt.scatter('area', 'poptotal', 

                data=midwest.loc[midwest.category==category, :], 

                s=20, c=colors[i], label=str(category))

# Decorations

plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000),

              xlabel='Area', ylabel='Population')



plt.xticks(fontsize=12); plt.yticks(fontsize=12)

plt.title("Scatterplot of Midwest Area vs Population", fontsize=22)

plt.legend(fontsize=12)    

plt.show()    
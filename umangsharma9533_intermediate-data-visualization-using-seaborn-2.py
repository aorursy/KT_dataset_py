# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
insurance_df=pd.read_csv('/kaggle/input/dataforseabornintermediatenb/insurance_premiums.csv')
# Create the same PairGrid but map a histogram on the diag

g = sns.PairGrid(insurance_df,vars=["fatal_collisions", "premiums"])

g2 = g.map_diag(plt.hist)

g3 = g2.map_offdiag(plt.scatter)



plt.show()

plt.clf()
# Create a pairwise plot of the variables using a scatter plot

sns.pairplot(data=insurance_df,

        vars=["fatal_collisions", "premiums"],

        kind='scatter')



plt.show()

plt.clf()
# Plot the same data but use a different color palette and color code by Region

sns.pairplot(data=insurance_df,

        vars=["fatal_collisions", "premiums"],

        kind='scatter',

             diag_kind='hist',

        hue='Region',

        palette='RdBu',

        diag_kws={'alpha':.5})



plt.show()

plt.clf()
# Build a pairplot with different x and y variables

sns.pairplot(data=insurance_df,

        x_vars=["fatal_collisions_speeding", "fatal_collisions_alc"],

        y_vars=['premiums', 'insurance_losses'],

        kind='scatter',

        hue='Region',

        palette='husl')



plt.show()

plt.clf()
# plot relationships between insurance_losses and premiums using kde for diag kind and reg for other two use region column as hue to understand difference

sns.pairplot(data=insurance_df,

             vars=["insurance_losses", "premiums"],

             kind='reg',

             palette='BrBG',

             diag_kind = 'kde',

             hue='Region')



plt.show()

plt.clf()
bike_rental_df=pd.read_csv('/kaggle/input/dataforseabornintermediatenb/bike_share.csv')
# Build a JointGrid comparing humidity and total_rentals

sns.set_style("whitegrid")

g = sns.JointGrid(x="hum",

            y="total_rentals",

            data=bike_rental_df,

            xlim=(0.1, 1.0)) 



g.plot(sns.regplot, sns.distplot)



plt.show()

plt.clf()
# Create a jointplot similar to the JointGrid 

sns.jointplot(x="hum",

        y="total_rentals",

        kind='reg',

        data=bike_rental_df)



plt.show()

plt.clf()
# Plot temp vs. total_rentals as a regression plot wirh with a 2nd order polynomial regression

sns.jointplot(x="temp",

         y="total_rentals",

         kind='reg',

         data=bike_rental_df,

         order=2,

         xlim=(0, 1))



plt.show()

plt.clf()
# Plot a jointplot showing the residuals

sns.jointplot(x="temp",

        y="total_rentals",

        kind='resid',

        data=bike_rental_df,

        order=2)



plt.show()

plt.clf()
#Complex joint

# Create a jointplot of temp vs. casual riders

# Include a kdeplot over the scatter plot

g = (sns.jointplot(x="temp",

             y="casual",

             kind='scatter',

             data=bike_rental_df,

             marginal_kws=dict(bins=10, rug=True))

    .plot_joint(sns.kdeplot))

    

plt.show()

plt.clf()
# Replicate the above plot but only for registered riders

g = (sns.jointplot(x="temp",

             y="registered",

             kind='scatter',

             data=bike_rental_df,

             marginal_kws=dict(bins=10, rug=True))

    .plot_joint(sns.kdeplot))



plt.show()

plt.clf()
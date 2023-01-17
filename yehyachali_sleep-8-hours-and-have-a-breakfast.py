import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from itertools import permutations

import plotly.express as px
sleep = pd.read_csv("../input/sleepstudypilot/SleepStudyData.csv")

sleep
# showing a count plot of each column

plt.figure(figsize=(30,20))

for i, c in enumerate(sleep.columns):

    plt.subplot(2,3,i+1)

    sns.countplot(sleep[c])

plt.show()    
# geting all the permutations with two columns

column_permutations = permutations(sleep.columns,2)



# visualizing the obtained permutations (count plot on the first column with different colors for each category)

plt.figure(figsize=(30,30))

for i, c in enumerate(column_permutations):

    plt.subplot(6,5,i+1)

    plt.title(i+1)

    sns.countplot(sleep[c[0]], hue=sleep[c[1]])

plt.show()
# swarm plot with two columns

sns.catplot(x="Enough", y="Tired",

            hue="PhoneTime", col="PhoneReach",

            data=sleep, kind="swarm")

plt.show()
# visualizing boxplots grouped by Tired with different colors for PhoneTime and PhoneReach 

plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

sns.boxplot(x="Tired",y="Hours",hue="PhoneTime",data=sleep)

plt.subplot(1,2,2)

sns.boxplot(x="Tired",y="Hours",hue="PhoneReach",data=sleep)

plt.show()
# visualizing boxplots grouped by Enough with different colors for PhoneTime and PhoneReach 

plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

sns.boxplot(x="Enough",y="Hours",hue="PhoneTime",data=sleep)

plt.subplot(1,2,2)

sns.boxplot(x="Enough",y="Hours",hue="PhoneReach",data=sleep)

plt.show()
# swarm plot grouped by Tired with different colors for Breakfast

sns.swarmplot(x="Tired",y="Hours",data=sleep,hue='Breakfast')

plt.show()
# using plotly

# getting the mean over Tired grouped by Breakfast

fig = px.bar(sleep.groupby("Breakfast")["Tired"].mean().reset_index(), x='Breakfast', y='Tired')

fig.show()
# using plotly

# getting the mean over Hours grouped by Tired

fig = px.bar(sleep.groupby("Tired")["Hours"].mean().reset_index(), x='Tired', y='Hours', color="Tired", title="the mean of hours slept per tiredness")

fig.show()
# ploting the relation between being tired and hours of sleep with different colors for Breakfast 

sns.relplot(x="Hours", y="Tired", ci=None, kind="line", hue="Breakfast", data=sleep)

plt.show()
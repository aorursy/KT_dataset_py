# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
crime_data = pd.read_csv("../input/database.csv")
#Lets see what the data looks like

crime_data.head(50)
#I am more interested in knowing what all different kinds of crimes actually exist

crime_data["Crime Type"].unique()
# So, we just have 2 kinds of crime - ['Murder or Manslaughter', 'Manslaughter by Negligence']

# So, kind of crime does not give so much information. Let's look at Weapon

crime_data["Weapon"].unique()

# What would be more interesting is to find information of city wise 
# Wouldn't it be more interesting to get counts of these crime 

from collections import Counter

Counter(crime_data["Weapon"].values)
# Weapon availability is critical for crime, lets see statewise pie crime counts given weapon is Handgun

handgun_state= dict(Counter(crime_data[crime_data["Weapon"]=="Handgun"]["State"].values))
# let us plot this information into a pie chart 

import matplotlib.pyplot as plt

import seaborn as sns

ax = sns.countplot(x="State", hue="Weapon", data=crime_data[crime_data["Weapon"]=="Handgun"])

ax.set_xticklabels(handgun_state.keys(),rotation=90)
#Let's just look at all weapons 

plt.figure(figsize=(12, 6))

ax = sns.countplot(x="State", hue="Weapon", data=crime_data)

ax.set_xticklabels(handgun_state.keys(),rotation=90)

ax.label_outer
# Killing unknown people seems like something killers would do. Let's see what weapons they use statewise

plt.figure(figsize=(12, 6))

#Lets see what the data looks like

ax = sns.countplot(x="State", hue="Weapon", data=crime_data[crime_data["Relationship"]=="Unknown"])

ax.set_xticklabels(handgun_state.keys(),rotation=90)

ax.label_outer
crime_data.columns
# Removing columns which can't be used for analysis

crime_data_filter1 = crime_data[['City',

       'State', 'Year', 'Month', 'Incident', 'Crime Type', 'Crime Solved',

       'Victim Sex', 'Victim Age', 'Victim Race', 'Victim Ethnicity',

       'Perpetrator Sex', 'Perpetrator Age', 'Perpetrator Race',

       'Perpetrator Ethnicity', 'Relationship', 'Weapon', 'Victim Count',

       'Perpetrator Count', 'Record Source']]
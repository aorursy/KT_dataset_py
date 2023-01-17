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
df = pd.read_csv("../input/KaggleV2-May-2016.csv", index_col='AppointmentID')

df.head()
from matplotlib import pyplot as plt

import seaborn as sns



sns.set(style="darkgrid")



# 1. Enlarge the plot

plt.figure(figsize=(12,24))



# Count Plot (a.k.a. Bar Plot)

sns.countplot(y='Neighbourhood', data=df)



# Rotate y-labels

plt.yticks(rotation=30)



# Don't forget the title :)

plt.title("Number of patients with an appointment in specific neighbourhoods")

plt.show()
# from http://seaborn.pydata.org/tutorial/categorical.html

sns.set(style="ticks", color_codes=True)



g = sns.FacetGrid(df, col="No-show",  row="Gender")

g = g.map(plt.hist, "Age")

plt.show()
df_show = df['No-show']

df_alcohol = df['Alcoholism']



#descriptives

obs = pd.crosstab(df_show, df_alcohol)

obs_freq = pd.crosstab(df_show, df_alcohol, normalize='columns')



print(obs)

print(obs_freq)

df_sms = df['SMS_received']



#descriptives

obs_freq = pd.crosstab(df_show, df_sms, normalize='columns')



print(obs_freq)



# Count Plot (a.k.a. Bar Plot)

plt.figure(figsize=(10,6))

sns.countplot(x='SMS_received', hue='No-show', data=df, palette='cool')

plt.title('SMS reminder and no-show rates among patients')

plt.show()
from scipy.stats import chi2_contingency



#two-way chi-square test 

obs = pd.crosstab(df_show, df_sms)



chisq, p, _, exp = chi2_contingency(obs)



print('Chi-square score is %f, p value is %f' % (chisq, p))

print('Observed values')

print(obs)

print('Expected values')

print(exp)
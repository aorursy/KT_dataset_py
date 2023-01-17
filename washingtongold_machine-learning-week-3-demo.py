# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

data.head()
data.columns
us = data[data['Country/Region'] == 'US']

us
us = us.drop(['Province/State','Country/Region','Lat','Long'],axis=1)
us
us.T
us = us.T.reset_index()

us
us = us.rename(columns={'index':'date',225:'confirmed'})

us
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(18,5)) #step 1: coordinates/figure



sns.barplot(x='date',y='confirmed',data=us) #step 2: specify type and step 3: data



plt.show() #show the plot by itself
plt.figure(figsize=(18,5)) #step 1: coordinates/figure



sns.set_style('whitegrid') #step 1: coordinates/figure



sns.barplot(x='date',y='confirmed',data=us) #step 2: specify type and step 3: data



plt.show() #show the plot by itself
plt.figure(figsize=(18,5)) #step 1: coordinates/figure

sns.set_style('whitegrid') #step 1: coordinates/figure

sns.barplot(x='date',y='confirmed',data=us) #step 2: specify type and step 3: data

plt.xticks(rotation=90) #step 1: coordinates/figure

plt.show() #show the plot by itself
plt.figure(figsize=(18,5)) #step 1: coordinates/figure

sns.set_style('whitegrid') #step 1: coordinates/figure

sns.lineplot(x='date',y='confirmed',data=us) #step 2: specify type and step 3: data

plt.xticks(rotation=90) #step 1: coordinates/figure

plt.show() #show the plot by itself
plt.figure(figsize=(18,5)) #step 1: coordinates/figure

sns.set_style('whitegrid') #step 1: coordinates/figure

sns.scatterplot(x='date',y='confirmed',data=us) #step 2: specify type and step 3: data

plt.xticks(rotation=90) #step 1: coordinates/figure

plt.show() #show the plot by itself
italy = data[data['Country/Region']=='Italy']

italy
italy = italy.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

italy
italy = italy.T

italy
italy = italy.reset_index()

italy
italy = italy.rename(columns={'index':'date',137:'confirmed'})

italy
plt.figure(figsize=(18,5)) #step 1: coordinates/figure

sns.set_style('whitegrid') #step 1: coordinates/figure



#United States data

sns.scatterplot(x='date',y='confirmed',data=us) #step 2 and 3



#Italy data

sns.scatterplot(x='date',y='confirmed',data=italy) #step 2 and 3



plt.xticks(rotation=90) #step 1: coordinates/figure

plt.show() #show the plot by itself
plt.figure(figsize=(18,5)) #step 1: coordinates/figure

sns.set_style('whitegrid') #step 1: coordinates/figure



#United States data

sns.scatterplot(x='date',y='confirmed',data=us,label='US') #step 2 and 3



#Italy data

sns.scatterplot(x='date',y='confirmed',data=italy,label='Italy') #step 2 and 3



plt.legend() #display the legend

plt.xticks(rotation=90) #step 1: coordinates/figure

plt.show() #show the plot by itself
covid = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

covid.head()
covid.columns
sub_data = covid[['age','outcome']]

sub_data
sub_data = sub_data.dropna()

sub_data
sub_data['outcome'].unique()
def process_outcome(x):

    if x=='discharged' or x=='discharge' or x=='Discharged':

        return 'Discharged'

    elif x=='died' or x=='death' or x=='severe':

        return 'Death/Severe'

    elif x=='stable' or x == 'recovered':

        return 'Stable'

    else:

        return np.nan

sub_data['outcome'] = sub_data['outcome'].apply(process_outcome)
sub_data = sub_data.dropna()

sub_data
sub_data['age'].apply(type)
sub_data['age'] = sub_data['age'].apply(int)
sub_data['age'].unique()
def process_age(age):

    if len(age.split('-'))==2: #if it is a range, e.g. '70-79'.split() -> ['70','79']

        return (float(age.split('-')[0]) + float(age.split('-')[1]))/2

        #return the average of the two bounds

    else:

        return float(age)

        #otherwise, return the float of the age

        

sub_data['age'] = sub_data['age'].apply(process_age)
plt.figure(figsize=(10,5))

sns.distplot(sub_data['age'])
discharged = sub_data[sub_data['outcome']=='Discharged']

death = sub_data[sub_data['outcome']=='Death/Severe']

stable = sub_data[sub_data['outcome']=='Stable']
plt.figure(figsize=(15,6)) #create figure and specify size



sns.distplot(discharged['age'],label='Discharged') #plot discharged

sns.distplot(death['age'],label='Death') #plot death/severe

sns.distplot(stable['age'],label='Stable') #plot stable



plt.legend() #display the legend so we know which distributions are which

plt.title('Ages of People by Outcome') #Add a title

plt.show() #show plot
plt.figure(figsize=(15,6)) #create figure + specify size



sns.boxplot(x='age',y='outcome',data=sub_data)



plt.title('Ages of People by Outcome')

plt.show()
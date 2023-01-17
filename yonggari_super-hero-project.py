# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





info = pd.read_csv('../input/superhero-set/heroes_information.csv')

powerful = pd.read_csv('../input/superhero-set/super_hero_powers.csv')
#SEE IF NaN VALUES EXISTS

print(info.isna().any())

print(powerful.isna().any())



#ADDS UP ALL THE NULL VALUES

print(info.isnull().sum())

print(powerful.isnull().sum())



#REPLACE THE '-' WITH 'N/A'

info.replace(to_replace='-',value='N/A',inplace=True)

info['Publisher'].fillna('N/A',inplace=True)



#DROP THE 'UNNAMED COLUMN' IN HERO_INFO CSV

info.drop('Unnamed: 0',axis=1,inplace=True)



#REPLACES NEGATIVE VALUE OF HEIGHT AND WEIGHT WITH 'N/A'

info.replace(-99.0, np.nan, inplace=True)









print(info.shape)

print(info.info())

print(info.head())

print(info.dtypes)









#PUBLISHER BREAKDOWN

#PRINTS OUT THE NUMBER FOR EACH PUBLISHER

print(info['Publisher'].value_counts())



#PRINTS OUT THE TOTAL NUMBER OF PUBLISHER COUNT

print(info.Publisher.count())



#BAR GRAPH OF PUBLISHER BY COUNT

fig = plt.figure(figsize=(12,7))

fig.add_subplot(1,1,1)

sns.countplot(x='Publisher',data=info)

plt.xticks(rotation=70)

plt.tight_layout()

plt.show()



#PIE CHART OF MARVEL, DC AND OTHER PUBLISHERS

labels = 'Marvel', 'DC', 'Others'

sizes = [388, 215, 131]

explode = (0.1, 0, 0 )



fig1 , ax1 = plt.subplots()



ax1.pie(sizes,

        explode = explode,

        labels = labels,

        autopct = '%1.1f%%',

        shadow = True,

        startangle = 100)

ax1.axis ('equal')

plt.show()

















#ALIGNMENT BREAKDOWN

print(info['Alignment'].value_counts())















#GENDER BREAKDOWN

#BOX PLOT OF HEIGHT & WEIGHT DISTRIBUTION BY GENDER

fig=plt.figure(figsize=(14,8))

fig.add_subplot(1,2,1)

sns.boxplot(x='Gender',y='Weight',data=info)

fig.add_subplot(1,2,2)

sns.boxplot(x='Gender',y='Height',data=info)

plt.show()





#BAR GRAPH OF GENDER COUNT FOR MARVEL HEROES

sns.countplot(info['Gender'][info['Publisher']=='Marvel Comics'])

plt.title('Gender count - Marvel Comics')



#BAR GRAPH OF GENDER COUNT FOR DC HEROES

sns.countplot(info['Gender'][info['Publisher']=='DC Comics'])

plt.title('Gender Count - DC comics')





#BAR GRAPH OF MALE/FEMALE SUPERHEROS

info_gender = info['Gender'].value_counts().head()

trace = go.Bar(

    y=info_gender.index[::-1],

    x=info_gender.values[::-1],

    orientation = 'h',

    marker=dict(

        color=info_gender.values[::-1]

    ),

)



layout = dict(

    title='Gender Distribution of Superheroes',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Superheroes")

plot(fig)

#%%



#POWER BREAKDOWN



#SEE IF NaN VALUES EXISTS

print(powerful.isna().any())



#THIS ONE ADDS UP ALL THE NULL VALUES

print(powerful.isnull().sum())



#TURNS TRUE/FALSE TO A "0" & "1" FORMATION

power=powerful*1











print(power.shape)

print(power.info())

print(power.head())

print(power.dtypes)











#ADDS ALL THE '1' IN THE ROW AND GIVES US A SUM FOR EACH CHARACTER

power.loc[:, 'no_of_powers'] = power.iloc[:, 1:].sum(axis=1)









#GETTING A TABLE OF NAME TO POWER NUMBER FROM MOST TO LEAST

most_powers=power[['hero_names','no_of_powers']]

most_powers=most_powers.sort_values('no_of_powers',ascending=False)



#GIVES THE TOP 10 HERO NAMES WITH MOST POWER

print(most_powers.head(10))







print(np.mean(most_powers.no_of_powers))

print(np.median(most_powers.no_of_powers))









#BAR GRAPH OF THE TOP 20 SUPERHERO POWERS. NUMBER OF POWERS BY NAME OF SUPERHERO

fig, ax = plt.subplots()



fig.set_size_inches(13.7, 10.27)



sns.set_context("paper", font_scale=1.5)

f=sns.barplot(x=most_powers["hero_names"].head(20), y=most_powers['no_of_powers'].head(20), data=most_powers)

f.set_xlabel("Name of Superhero",fontsize=18)

f.set_ylabel("No. of Superpowers",fontsize=18)

f.set_title('Top 20 Superheroes having highest no. powers')

for item in f.get_xticklabels():

    item.set_rotation(90)

    

    

    

    

    

#PRINTS HOW MANY CAN FLY

print(len(power[(power['Flight'] == 1)]))
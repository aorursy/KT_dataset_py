# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read using encoding="ISO-8859-1", otherwise won't work



df=pd.read_csv('../input/crimes-in-boston/crime.csv',index_col=0, encoding="ISO-8859-1")

df.head()
#Check what are the unique values in Offense Code Group

df.OFFENSE_CODE_GROUP.unique()
#Check how many counts per unique value of OFFENSE_CODE_GROUP



a=df.OFFENSE_CODE_GROUP.value_counts()

print(a)
# Set the width and height of the figure

plt.figure(figsize=(13,5))



# Add title

plt.title("Number of crimes in 2015-2018")



sns.barplot(x=a.index, y=a)



#Rotate x-labels, otherwise it's utterly hectic

plt.xticks(rotation=90)



# Add label for vertical axis

plt.ylabel("Counts")
# Let's group by year

df_years=df.groupby('YEAR')['OFFENSE_CODE_GROUP']

df_years.first()

#Number of unique crimes by year, sorted in descending order



df_2015=df_years.get_group(2015).value_counts()

df_2016=df_years.get_group(2016).value_counts()

df_2017=df_years.get_group(2017).value_counts()

df_2018=df_years.get_group(2018).value_counts()
plt.figure(figsize=(13,5))



# Add title

plt.title("Number of crimes in 2015")



sns.barplot(x=df_2015.index, y=df_2015)

plt.xticks(rotation=90)

# Add label for vertical axis

plt.ylabel("Counts")
# Apparently Boston seems to be a dangerous city to drive in. 
# Code 3303 refers to NOISY PARTY/RADIO-ARREST. Let's see how loud partying in Boston was like throughout the years



df_codes=df.groupby('YEAR')['OFFENSE_CODE']

df_codes.first()



#Number of 3303 codes grouped by year



df_code2015=df_codes.get_group(2015).value_counts()

df_code2016=df_codes.get_group(2016).value_counts()

df_code2017=df_codes.get_group(2017).value_counts()

df_code2018=df_codes.get_group(2018).value_counts()
party_2015=df_code2015[3303]

party_2016=df_code2016[3303]

party_2017=df_code2017[3303]

party_2018=df_code2018[3303]

#Creating a dataframe using dictionary

data={'Year':[2015, 2016, 2017, 2018], 'Noisy parties':[party_2015, party_2016, party_2017, party_2018]}

noisy_parties=pd.DataFrame(data)

noisy_parties
plt.figure(figsize=(4,4))



# Add title

plt.title("Number of noisy parties per year in Boston")



sns.barplot(x=noisy_parties.Year, y=noisy_parties['Noisy parties'])

plt.xticks(rotation=30)

# Add label for vertical axis

plt.ylabel("No of Parties")
#2016 was a good year for noisy partying. 
#Name of streets where crimes had been commited, sorted by descending order

df.STREET.value_counts()


from wordcloud import WordCloud, STOPWORDS



#texto=list(df.OFFENSE_CODE_GROUP)

texto=",".join(list(df.OFFENSE_CODE_GROUP))



catcloud = WordCloud(

    stopwords=None,

    background_color='black',

    width=1200,

    height=800

).generate(texto)



plt.imshow(catcloud, alpha=0.8)

plt.axis('off')

plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import plotly.plotly as py
from matplotlib import rc
#Let's write a function to read an excel sheet and do what needs to be done :)
def read_sheet(name):
    composition = pd.read_excel('../input/Gallup2013.xlsx',sheet_name=name)
    composition = composition[:-1]
    composition = composition.T
    composition.columns = composition.iloc[0]
    composition = composition[1:]
    return composition

r = [0,1,2,3]
age_composition = read_sheet('AGE COMPOSITION OF PARTY VOTES ')
greenBars = age_composition['New voters (Age 18-24)']
orangeBars = age_composition['Age 25 – 29']
blueBars = age_composition['Age 30 – 34']
redBars = age_composition['Age 35 - 49']
yellowBars = age_composition['50 +']

 
# plot
barWidth = 0.85
names = ('All Pakistan','PML(N)','PTI','PPP')
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="Age 18-24")
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label="Age 25-29")
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth, label="Age 30-35")

plt.bar(r, redBars, bottom=[i+j+k for i,j,k in zip(greenBars, orangeBars,blueBars)], color='#ff0000', edgecolor='white', width=barWidth,label="Age 35-49")
 

plt.bar(r, yellowBars, bottom=[i+j+k+l for i,j,k,l in zip(greenBars, orangeBars,blueBars,redBars)], color='#ffff00', edgecolor='white', width=barWidth,label="Age 50+")
# Custom x axis
plt.xticks(r, names)
plt.xlabel("Party")
plt.ylabel("Percentage")
plt.title("Age Composition")
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
 
# Show graphic

plt.show()

r = [0,1,2,3]
education_composition = read_sheet('EDUCATIONAL COMPOSITION OF PART')
greenBars = education_composition['Illiterate']
orangeBars = education_composition['Up to Middle School']
blueBars = education_composition['High School and Intermediate']
redBars = education_composition['Bachelors and Masters (College)']
 
# plot
barWidth = 0.85
names = ('All Pakistan','PML(N)','PTI','PPP')
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="Illiterate")
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label="Age Up to Middle School")
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth, label="High School and Intermediate")

plt.bar(r, redBars, bottom=[i+j+k for i,j,k in zip(greenBars, orangeBars,blueBars)], color='#ff0000', edgecolor='white', width=barWidth,label="Bachelors and Masters (College)")
 
# Custom x axis
plt.xticks(r, names)
plt.xlabel("Party")
plt.ylabel("Percentage")
plt.title("Education Composition")
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
 
# Show graphic

plt.show()
r = [0,1,2,3]
income_composition = read_sheet('INCOME COMPOSITION OF PARTY VOT')
greenBars = income_composition['Upto  Rs.7,000']
orangeBars = income_composition['Rs 7,001 - 10,000']
blueBars = income_composition['Rs 10,001 - 15,000']
redBars = income_composition['Rs 15,000 - Rs. 30,000']
yellowBars = income_composition['More than Rs. 30,000']

 
# plot
barWidth = 0.85
names = ('All Pakistan','PML(N)','PTI','PPP')
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="Upto  Rs.7,000")
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label="Rs 7,001 - 10,000")
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth, label="Rs 10,001 - 15,000")

plt.bar(r, redBars, bottom=[i+j+k for i,j,k in zip(greenBars, orangeBars,blueBars)], color='#ff0000', edgecolor='white', width=barWidth,label="Rs 15,000 - Rs. 30,000")
 

plt.bar(r, yellowBars, bottom=[i+j+k+l for i,j,k,l in zip(greenBars, orangeBars,blueBars,redBars)], color='#ffff00', edgecolor='white', width=barWidth,label="More than Rs. 30,000")
# Custom x axis
plt.xticks(r, names)
plt.xlabel("Party")
plt.ylabel("Percentage")
plt.title("Income Composition")
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
 
# Show graphic

plt.show()

r = [0,1,2,3]
gender_composition = read_sheet('GENDER COMPOSITION OF PARTY VOT')
greenBars = gender_composition['Men']
orangeBars = gender_composition['Women']

 
# plot
barWidth = 0.85
names = ('All Pakistan','PML(N)','PTI','PPP')
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="Men")
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label="Women")

 
# Custom x axis
plt.xticks(r, names)
plt.xlabel("Party")
plt.ylabel("Percentage")
plt.title("Gender Composition")
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
 
# Show graphic

plt.show()
voting_reason = pd.read_excel('../input/Gallup2013.xlsx',sheet_name='Voting Reason (Gallup Exit Poll')
voting_reason.drop(['Percentage of Respondents (1993)', 'Percentage of Respondents (1997)'], axis=1,inplace=True)
voting_reason = voting_reason.T
cols = voting_reason.iloc[0]
voting_reason = voting_reason.iloc[1:]
voting_reason.columns = cols

r = [0,1]
greenBars = voting_reason['Party loyal']
orangeBars = voting_reason['Development seekers']
blueBars = voting_reason['Patronage seekers']
redBars = voting_reason['Legislation minded']
yellowBars = voting_reason['Value/Morality Seekers']
purpleBars = voting_reason['Biradri bound']
blackBars = voting_reason['Skeptics']
pinkBars = voting_reason["Don't know"]
 
# plot
barWidth = 0.85
names = ('2013','2008')
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="Party loyal")
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label="Development seekers")
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth, label="Patronage seekers")

plt.bar(r, redBars, bottom=[i+j+k for i,j,k in zip(greenBars, orangeBars,blueBars)], color='#ff0000', edgecolor='white', width=barWidth,label="Legislation minded")

plt.bar(r, yellowBars, bottom=[i+j+k+l for i,j,k,l in zip(greenBars, orangeBars,blueBars,redBars)], color='#ffff00', edgecolor='white', width=barWidth,label="Value/Morality Seekers")
 
plt.bar(r, purpleBars, bottom=[i+j+k+l+m for i,j,k,l,m in zip(greenBars, orangeBars,blueBars,redBars,yellowBars)], color='#800080', edgecolor='white', width=barWidth,label="Biradri bound")

plt.bar(r, blackBars, bottom=[i+j+k+l+m+n for i,j,k,l,m,n in zip(greenBars, orangeBars,blueBars,redBars,yellowBars,purpleBars)], color='#000000', edgecolor='white', width=barWidth,label="Skeptics)")

plt.bar(r, pinkBars, bottom=[i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(greenBars, orangeBars,blueBars,redBars,yellowBars,purpleBars,blackBars)], color='#ff69b4', edgecolor='white', width=barWidth,label="Don't know")

# Custom x axis
plt.xticks(r, names)
plt.xlabel("year")
plt.ylabel("percentage")
plt.title("Voting Reason")
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
 
# Show graphic

plt.show()

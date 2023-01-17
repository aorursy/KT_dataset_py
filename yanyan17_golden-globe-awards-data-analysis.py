# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import squarify



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

gga = pd.read_csv("../input/golden-globe-awards/golden_globe_awards.csv")
# Dimensions of the dataset

display(gga.shape)

display(gga.columns)

display(gga.head())
#In line4, the film data shows "NaN"

#Check the Na values in the dataset and figure out where they come from 

gga.isnull().sum()

gga[gga.isnull().any(axis=1)]
#Replace the NaN values with nominees' names and check the null values again.

gga['film'] = np.where(gga['film'].isnull(), gga['nominee'], gga['film'] )

gga.isnull().sum()

gga.sample(10)
#Top 10 highest nominated films 

top_10_nominee_film=gga['film'].value_counts().head(n=10).reset_index()

top_10_nominee_film.columns =['film', 'nomination'] 

top_10_nominee_film

#Create a new dataset called gga_winner to store the data where win==True

gga_winner=gga[gga.win==True]

gga_winner.sample(10)
#Top 10 winning categories

top_10_category=gga_winner['category'].value_counts().head(n=10).reset_index()

top_10_category.columns =['category', 'win'] 

top_10_category



#Top 15 winning films

top_15_film=gga_winner['film'].value_counts().head(n=15).reset_index()

top_15_film.columns =['film', 'win'] 

top_15_film



#Top 20 nominees won most awards

top_20_nominee=gga_winner['nominee'].value_counts().head(n=20).reset_index()

top_20_nominee.columns =['nominee_name', 'win'] 

top_20_nominee



#Visualization 

#Barchart for top 15 films have most awards

plt.figure(figsize=(12,10))

sns.set_style('darkgrid')

sns.barplot(x='film', y='win', data=top_15_film, palette='hls')

plt.title('Top 15 films have most awards for Golden Globe Awards from 1944 to 2020', fontsize=12,fontweight="bold")

plt.xlabel('Film Name', fontsize=15)

plt.ylabel('Total Wins', fontsize=15)

plt.xticks(fontsize=13, rotation=90)

plt.yticks(fontsize=13)

plt.show()
#Treemap for Top 10 winning categories

#The darker the color and the bigger the square size, the more awards 

norm = matplotlib.colors.Normalize(vmin=min(top_10_category.win), vmax=max(top_10_category.win))

colors = [matplotlib.cm.Reds(norm(value)) for value in top_10_category.win]



plt.figure(figsize=(18,18))

squarify.plot(label=top_10_category.category,sizes=top_10_category.win, color = colors, alpha=.7 )

plt.title("Top 10 winning categories for Golden Globe Awards from 1944 to 2020",fontsize=18,fontweight="bold")



plt.axis('off')

plt.show()
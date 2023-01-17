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



nx = pd.read_csv('../input/netflix-shows/netflix_titles.csv')







#DROPS THE ROWS WITH NaN VALUE

nex = nx.dropna()

#RESETS THE INDEX NUMBER

nex = nex.reset_index(drop=True)



print(nex.head())

print(nex.describe())









#BREAKDOWN OF TYPE

print(nex['type'].value_counts())

labels = 'Movies', 'TV Shows'

sizes = [3678, 96]

explode = (0,0.1)

plt.pie(sizes, labels=labels,shadow = True, startangle=90, explode = explode, autopct='%1.1f%%')

plt.show()









#BREAKDOWN OF DIRECTOR

#NUMBER OF DIRECTORS

print(nex.director.count())

#NUMBER OF DIFFERENT DIRECTORS

print(nex.director.nunique())

#PRINTS DIRECTOR COUNT NUMBERS (EG. HOW MANY MOVIES EACH DIRECTOR HAS)

print(nex['director'].value_counts())









#BREAKDOWN OF COUNTRY

print(nex['country'].value_counts())

n_1_country = 0

n_2_country = 0

#SEPARATES THE SINGULAR COUNTRIES WITH THE MULTIPLE COUNTRIES

column_data = list(nex['country'].values)

for i in range(0, len(column_data)):

	temporary_data = column_data[i]

	count = len(temporary_data.split(','))

	if count == 1:

		n_1_country += 1

	elif count >= 2:

		n_2_country += 1

	else:

		print('Some issues spotted.')

#PRINTS HOW MANY INDIVIDUAL COUNTRIES

print(n_1_country)

#PRINTS HOW MANY MULTIPLE COUNTRIES

print(n_2_country)



labels = 'Single Countries', 'Multiple Countries'

sizes = [3104, 670]

explode = (0,0.1)

plt.pie(sizes, labels=labels,shadow = True, startangle=90, explode = explode, autopct='%1.1f%%')

plt.show()









#DATE ADDED BREAKDOWN

print(nex.date_added.min())

print(nex.date_added.max())

print(nex.date_added.nunique())



unique_date = list( set(nex['date_added'].values))

print(unique_date)



counter = []



for date in unique_date:

	pandas_slice = nex[nex['date_added'] == date]

	counter.append([date, len(pandas_slice)])



counter = pd.DataFrame(counter)

counter.columns = ['date_added', 'Number of Movies']









#RELEASE YEAR

print(nex.release_year.describe())

print(nex['release_year'].value_counts())





print(nex.release_year.sort_values())



ryc = (nex['release_year'].value_counts())

print(ryc.max())

print(ryc.min())









#RATING BREAKDOWN

print(nex.rating.describe())

print(nex['rating'].value_counts())



labels = 'TV-MA', 'TV-14', 'R', 'TV-PG', 'PG-13', 'PG'

sizes = [1189,917,501,358,278,176]

explode = (0.1,0.1,0,0,0,0)

plt.pie(sizes, labels=labels,shadow = True, startangle=90, explode = explode, autopct='%1.1f%%')

plt.show()











#DURATION BREAKDOWN

print(nex.duration.describe())

print(nex['duration'].value_counts())

print(nex.duration.nunique())
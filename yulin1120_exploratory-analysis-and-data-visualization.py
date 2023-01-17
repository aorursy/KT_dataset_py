# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
restaurant = pd.read_csv('../input/restaurant-and-food-inspections-alaska-usa/Restaurant_and_Food_Inspection_Alaska.csv')
restaurant.info()
restaurant.describe()
cities=restaurant['business_city'].value_counts()



fig = px.pie(cities,

             values=cities.values,

             names=cities.index,

             title="Restaurant Distribution",

             template="seaborn")

fig.update_traces(rotation = -70, textinfo="label+percent")

fig.show()
restaurant['violation'] = 1

restaurant.loc[restaurant['violation_code'].isna(),'violation'] = 0



plt.figure(figsize=(11,8))

sns.countplot(x='violation',data=restaurant)

plt.title('Violation',size=20)
#violation each year



restaurant['inspection_year'] = restaurant['inspection_date'].str.split('/').str[2]

restaurant['inspection_month'] = restaurant['inspection_date'].str.split('/').str[0]



plt.figure(figsize=(11,8))

sns.countplot(x= 'inspection_year', hue = 'violation',data=restaurant)

plt.title('Violation by Year',size=20)
#violation each month

plt.figure(figsize=(11,8))

sns.countplot(x= 'inspection_month', hue = 'violation',data=restaurant)

plt.title('Violation by Month',size=20)
plt.figure(figsize=(11,8))

sns.countplot(y='inspection_type', hue = 'violation',data=restaurant)

plt.title('Inspection Type by Violation Situation',size=20)
#violation score distribution by violation

plt.figure(figsize=(11,8))

sns.distplot(restaurant[restaurant['violation']==0]['inspection_score'], kde =False, bins =30, hist_kws={'label':'0'})

sns.distplot(restaurant[restaurant['violation']==1]['inspection_score'], kde =False, bins =30, hist_kws={'label':'1'})

plt.title('Inspection Score by Violation Situation',size=20)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
codes=restaurant['violation_code'].value_counts().head(20)



fig = px.pie(codes,

             values=codes.values,

             names=codes.index,

             title="Top 20 Violations",

             template="seaborn")

fig.update_traces(textinfo="label+value")

fig.show()
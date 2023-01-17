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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Now lets read the museum dataset.

museum_path='../input/data-for-datavis'

museum_filepath='museum_visitors.csv'

museum_file=os.path.join(museum_path,museum_filepath)

museum_data=pd.read_csv(museum_file,index_col=['Date'],parse_dates=True)
museum_data.head(5)
# Lets have a close look at the visitor data for one of the museums

museum_data['Avila Adobe'][museum_data.index.year==2018]
plt.figure(figsize=(16,6))

plt.title('Trend of visitors in museum')

plt.xlabel('Date')

plt.ylabel('visitors count')

sns.lineplot(data=museum_data)
plt.figure(figsize=(14,7))

plt.title('Trend of visitors in Avila Adobe museum')

plt.xlabel('Date')

plt.ylabel('visitors count')

sns.lineplot(data=museum_data['Avila Adobe'][museum_data.index.year==2018])
file_dir='../input/data-for-datavis'

file_name='flight_delays.csv'

file=os.path.join(file_dir,file_name)

flight_data=pd.read_csv(file)

flight_data.head(12)
# The below line of code will give us the null values in the US column

flight_data[pd.isnull(flight_data.US)]
mean=flight_data['US'].mean()

flight_data.fillna(value=mean,inplace=True)
# Lets find the average arrival delay for spirit airlines flights. Code=NK

plt.figure(figsize=(16,6))

plt.title('Average Flight Delay for american airlines')

# plt.xticks(rotation='vertical')

plt.ylabel('months==>')

sns.heatmap(data=flight_data,annot=True)
file_dir='../input/data-for-datavis'

file_name='candy.csv'

file=os.path.join(file_dir,file_name)

candy_data=pd.read_csv(file)

candy_data.head()
# Which candybar has the highest winpercent

candy_data[candy_data['winpercent']==candy_data['winpercent'].max()]
# Which candybar has the highest percentage of sugar

candy_data[candy_data['sugarpercent']==candy_data['sugarpercent'].max()]
plt.figure(figsize=(16,8))

plt.title('Relationship between sugar and popularity of candybar')

sns.scatterplot(data=candy_data, x='sugarpercent', y='winpercent')
sns.regplot(data=candy_data, x='sugarpercent', y='winpercent')
sns.lmplot(data=candy_data, x='sugarpercent', y='winpercent', hue='chocolate')
# Let see if popularity is related to the pricing

sns.regplot(data=candy_data, x='pricepercent', y='winpercent')
sns.lmplot(data=candy_data, x='pricepercent', y='winpercent', hue='chocolate')
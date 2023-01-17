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

import seaborn as sns
dataset = pd.read_csv('../input/nutrition-facts/menu.csv')

dataset.head()
calories_data = dataset.iloc[:,0:5]

calories_data['Total Calories'] = calories_data['Calories']+calories_data['Calories from Fat']

calories_by_category = calories_data.groupby('Category').mean().sort_values(by='Total Calories')

calories_by_category
calories_by_category.iloc[:,:2].plot(kind='bar', stacked=True)

plt.title('Total Calories Average by Category')

plt.ylabel('Total Calories')

plt.show()
sandwich_data = dataset[dataset['Item'].str.contains('Sandwich')]

sandwich_data = sandwich_data.iloc[:,[1,6,8,11,13,15,17,20,21,22,23]]

sandwich_data['Total Daily Value'] = sandwich_data.sum(axis=1)

sandwich_data['Chicken Type'] = sandwich_data['Item'].str.extract("(Crispy|Grilled)")

sandwich_data['Item Type'] = sandwich_data['Item'].str.extract("(Classic|Club|Ranch|Bacon|Southern)")

sandwich_data
plt.figure(figsize=(10,7))

sns.barplot(sandwich_data['Item Type'], sandwich_data['Total Daily Value'], hue=sandwich_data['Chicken Type'])

plt.show()
plt.figure(figsize=(10,7))

sns.heatmap(sandwich_data.iloc[:,1:11], cmap='Blues', annot=True, 

            yticklabels=sandwich_data['Item Type']+sandwich_data['Chicken Type'])

plt.show()
egg_data = dataset[dataset['Item'].str.contains("with Egg")]

egg_data = egg_data[egg_data['Item'].str.contains("Sausage McMuffin|Sausage Biscuit")]

egg_data = egg_data.iloc[:,[1,6,8,11,13,15,17,20,21,22,23]]

egg_data['Total Daily Value'] = egg_data.sum(axis=1)

egg_data['Egg Type'] = egg_data['Item'].str.extract("(Egg Whites)")

egg_data['Egg Type'].fillna('Whole Egg', axis=0, inplace=True)

egg_data['Item Type'] = egg_data['Item'].str.extract("(Sausage McMuffin|Regular Biscuit|Large Biscuit)")

egg_data
plt.figure(figsize=(10,7))

sns.barplot(egg_data['Item Type'], egg_data['Total Daily Value'], hue=egg_data['Egg Type'])

plt.show()
plt.figure(figsize=(10,7))

sns.heatmap(egg_data.iloc[:,1:11], cmap='Blues', annot=True, 

            yticklabels=egg_data['Item'])

plt.show()
least_number_data = dataset.iloc[:,[1,6,8,11,13,15,17,20,21,22,23]]

least_number_data.head()
least_number_data.describe()
max_index=[]

for i in least_number_data.describe().columns:

    max_index.extend(least_number_data[least_number_data[i]==least_number_data.describe().loc['max',i]].index.tolist())
index_set = set(max_index)

index_list = list(index_set)
index_list
max_data = least_number_data.iloc[index_list,:]

max_data
from itertools import combinations



for i in combinations(max_data.index,6):

    table = least_number_data.iloc[list(i),:]

    l = []

    for n in max_data.columns[1:]:

        m = table[n].sum()

        l.append(m)

    if all(l>=100*np.ones(10)):

        print(i)

        break
minimum_menu_data = least_number_data.iloc[list(i),:]

minimum_menu_data
for s in least_number_data.describe().columns:

    print(s,': ', minimum_menu_data[s].sum())
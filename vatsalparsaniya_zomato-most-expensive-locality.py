# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
zomato_data = pd.read_csv('../input/zomato.csv',encoding='latin')
zomato_data.head()
zomato_data.info()
zomato_data['Country Code'].value_counts()
one_selected_country_dataset = zomato_data[zomato_data['Country Code']==1]

one_selected_country_dataset.head()
one_selected_country_dataset['Locality'].value_counts()
one_selected_country_dataset['Average Cost for two'].value_counts()
one_selected_country_dataset['Average Cost for two'] = one_selected_country_dataset['Average Cost for two'].astype(float)

locality_list = list(one_selected_country_dataset['Locality'].unique())

average_cost_of_2_by_locality=[]

for locality in locality_list:

    x = one_selected_country_dataset[one_selected_country_dataset['Locality']==locality]

    avg_cost = sum(x['Average Cost for two'])/len(x)

    average_cost_of_2_by_locality.append(avg_cost)

data = pd.DataFrame({'locality':locality_list,'Avg_food_cost_of_two':average_cost_of_2_by_locality})

new_index = (data['Avg_food_cost_of_two'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['locality'][:15],y=sorted_data['Avg_food_cost_of_two'][:15])

plt.xticks(rotation=90)

plt.xlabel("locality")

plt.ylabel("Food Cost of 2 people")

plt.title("country's most 15 expensive city's according Food of two(data by Zomato)")

plt.show()
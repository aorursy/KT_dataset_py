# Import the necessary libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Import the dataset

data = pd.read_csv('../input/zomato.csv')
# Get an idea about the data

data.head()
# Check the shape of the dataset

data.shape
# Check for null values in the dataset

pd.DataFrame({'Column':[i.upper() for i in data.columns],

                      'Count':data.isnull().sum().values,

                      'Percentage':((data.isnull().sum().values/len(data))*100).round(2)

                     })
data.rate = data.rate.str.replace('/5','')
locations = data.location.str.lower().unique()

print(f'There are {len(data.url.str.lower().unique())} restaurants across {len(locations)} locations in Bengaluru.')
unique_brands = data.name.unique()

print(f'{len(unique_brands)} brands are in the business.')
plt.figure(figsize=(20,3))

f = sns.countplot(x='listed_in(city)', data=data, order = data['listed_in(city)'].value_counts().index)

plt.xlabel('No.of Restaurants')

plt.ylabel('Locality')

f.set_xticklabels(f.get_xticklabels(), rotation=30, ha="right")

f
branches = data.groupby(['name']).size().to_frame('count').reset_index().sort_values(['count'],ascending=False)

fig = plt.figure(figsize=(20,4))

f = sns.barplot(x='name', y='count', data=branches[:10])

plt.xlabel('')

plt.ylabel('Branches')

f

print(f'{branches.iloc[0,0]} has the highest number of branches in the city')
cuisines = set()

for i in data['cuisines']:

    for j in str(i).split(', '):

        cuisines.add(j)

cuisines.remove('nan')
cuisines
print(f'There are {len(cuisines)} different types of cuisines available in Bengaluru')
locality = 'Banashankari'
# You can also pass null to display all the cuisines for the selected restaurants

cuisine = 'Bakery'
isOnline = 'Yes'

pd.DataFrame(data[['name', 'rate', 'approx_cost(for two people)']]

             [(data['location'].str.contains(locality)) 

              & (data['cuisines'].str.contains(cuisine)) 

              & (data['online_order'] == isOnline)]).sort_values(['rate'], ascending = False).drop_duplicates()
sns.countplot(x='book_table', data=data)

plt.xlabel('Booking available')

plt.ylabel('No.of Restaurants')
plt.figure(figsize=(15,4))

sns.countplot(x='listed_in(type)', data=data[data['location']==locality])

plt.xlabel('Restaurant Type')

plt.ylabel('No.of Restaurants')
dishes = {}

for i in data['dish_liked'][data['location']==locality]:

    for j in str(i).split(', '):

        if j in dishes.keys():

            dishes[j] = dishes[j] + 1

        else:

            dishes[j] = 1

_ = dishes.pop('nan')
pd.DataFrame.from_dict(dishes, orient='index', columns=['Count']).sort_values(['Count'], ascending=False)[:10]
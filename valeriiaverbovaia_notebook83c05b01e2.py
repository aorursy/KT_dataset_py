!pip install jovian opendatasets --upgrade --quiet
# getting url of a dataset
dataset_url = 'https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data' 
import opendatasets as od
od.download(dataset_url)
#backup
import pandas as pd 
airbnb_df=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
# directory of the file
data_dir = './new-york-city-airbnb-open-data'
import os
os.listdir(data_dir)
project_name = "airbnb_final" 
!pip install jovian --upgrade -q
import jovian
jovian.commit(project=project_name)
!pip install numpy pandas matplotlib seaborn --upgrade --quiet
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
airbnb_df=pd.read_csv('./new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airbnb_df
airbnb_df.info()
airbnb_df.last_review = pd.to_datetime(airbnb_df.last_review, errors='coerce')
airbnb_df.info()
airbnb_df.shape
airbnb_df.describe()
airbnb_df.drop(columns='host_name', inplace=True)
airbnb_df = airbnb_df[airbnb_df.price != 0]
airbnb_df = airbnb_df[airbnb_df.availability_365 != 0]
airbnb_df
not_reviewed = airbnb_df['reviews_per_month'].isna().sum()
print('A number of properties without any reviews is {}.'.format(int(not_reviewed)))
print('Available types of properties are:', airbnb_df['room_type'].unique())
print('Available areas of NY are: ', airbnb_df['neighbourhood_group'].unique())
total_reviews = airbnb_df['number_of_reviews'].sum()
print('A total number of reviews is {}.'.format(int(total_reviews)))
mean_reviews = airbnb_df['number_of_reviews'].mean()
print('The majotiry of properties have around {} reviews'.format(int(mean_reviews)))
most_expensive=airbnb_df.sort_values('price').tail(10)
most_expensive
cheapest = airbnb_df.sort_values('price').head(10)
cheapest
print('Number of properties in Manhattan:', airbnb_df[airbnb_df.neighbourhood_group=='Manhattan'].id.count())
print('Number of properties in Queens:', airbnb_df[airbnb_df.neighbourhood_group=='Queens'].id.count())
print('Number of properties in Brooklyn:', airbnb_df[airbnb_df.neighbourhood_group=='Brooklyn'].id.count())
print('Number of properties in Bronx:', airbnb_df[airbnb_df.neighbourhood_group=='Bronx'].id.count())
print('Number of properties in Staten Island:', airbnb_df[airbnb_df.neighbourhood_group=='Staten Island'].id.count())
location_counts_df = airbnb_df.groupby('neighbourhood_group')['neighbourhood'].count()
location_counts_df #counting the same as in the previous step but altogether in a single line
airbnb_df['min_sum'] = airbnb_df['minimum_nights'] * airbnb_df['price']
airbnb_df.sort_values('min_sum', ascending=True).head(5)
airbnb_df.sort_values('min_sum', ascending=True).tail(5)
jovian.commit()
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
location_counts_df
plt.hist(airbnb_df.neighbourhood_group, bins=5, rwidth=0.9);
plt.xticks(rotation=75)
plt.title('Neighbourhood Groups');

all_hoods = airbnb_df.neighbourhood_group.value_counts()
plt.pie(all_hoods, labels=all_hoods.index, autopct='%1.1f%%');
plt.hist(airbnb_df.room_type, bins=3, rwidth=0.9);
plt.title('Types of rooms')
plt.hist(airbnb_df.price, bins=np.arange(10,1000,10), color='purple')
plt.xlabel('Price per night')
plt.ylabel('Number of properties');
map_ny = sns.scatterplot(x=airbnb_df.longitude,y=airbnb_df.latitude,hue=airbnb_df.neighbourhood_group, size=airbnb_df.price, alpha=.3);
map_ny.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1);
from PIL import Image
img=Image.open('./new-york-city-airbnb-open-data/New_York_City_.png')
plt.imshow(img);
top_neighbourhood=airbnb_df.neighbourhood.value_counts().head(15)
sns.barplot(x=top_neighbourhood.index, y=top_neighbourhood)
plt.title('Most popular neighbourhoods')
plt.xticks(rotation=75);
top_neighbourhood
bottom_neighbourhood=airbnb_df.neighbourhood.value_counts().tail(15)
sns.barplot(x=bottom_neighbourhood.index, y=bottom_neighbourhood)
plt.title('Least popular neighbourhoods')
plt.xticks(rotation=75);
bottom_neighbourhood

appartments_df = airbnb_df[airbnb_df.room_type=='Entire home/apt'].copy()
appartments_df
map_ny_entire_place = sns.scatterplot(x=appartments_df.longitude,y=appartments_df.latitude,hue=appartments_df.neighbourhood_group, size=appartments_df.price, alpha=.3);
map_ny_entire_place.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1);
map_ny_entire_place = sns.scatterplot(x=appartments_df.longitude,y=appartments_df.latitude,hue=appartments_df.price);
map_ny_entire_place.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1);
appartments_maj_df = appartments_df[appartments_df.price<600].copy()
map_ny_entire_place_maj = sns.scatterplot(x=appartments_maj_df.longitude,y=appartments_maj_df.latitude,hue=appartments_maj_df.price);
map_ny_entire_place_maj.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1);
appartments_min_df = appartments_df[appartments_df.price<60].copy()
map_ny_entire_place_min = sns.scatterplot(x=appartments_min_df.longitude,y=appartments_min_df.latitude,hue=appartments_min_df.price);
map_ny_entire_place_min.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1);
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('A tale of 2 subplots')

ax1.plot(appartments_df.number_of_reviews, 'o-')
ax1.set_ylabel('Damped oscillation')

ax2.plot(appartments_df.price, '.-')
ax2.set_xlabel('time (s)')
ax2.set_ylabel('Undamped')

plt.show()
last_review_df=airbnb_df.last_review.value_counts()
last_review_df=last_review_df.sort_index()
last_review_df
plt.plot_date(last_review_df.index, last_review_df);
latest_reviews_df= airbnb_df[(airbnb_df['last_review'] > '2019-01-01')]
latest_reviews_df = latest_reviews_df.last_review.value_counts()
latest_reviews_df=latest_reviews_df.sort_index()
plt.xticks(rotation=70)
plt.plot_date(latest_reviews_df.index, latest_reviews_df);

import jovian
jovian.commit()
districts = airbnb_df['neighbourhood_group'].unique()
districts
appartments_min_df = appartments_df[appartments_df.price<60].copy()
district_df = airbnb_df[airbnb_df.neighbourhood_group == 'Bronx'].copy()
price = airbnb_df['price'].mean()
price
district_price={}
for district in districts:
    district_df = airbnb_df[airbnb_df.neighbourhood_group== district].copy()
    price = district_df['price'].mean()
    district_price[district]=price
print(district_price)
    
plt.bar(range(len(district_price)), list(district_price.values()), align='center')
plt.xticks(range(len(district_price)), list(district_price.keys()))

plt.show()

shared_df=airbnb_df[airbnb_df.room_type=='Shared room'].copy()
map_ny_shared = sns.scatterplot(x=shared_df.longitude,y=shared_df.latitude,hue=shared_df.price);
shared_expensive=shared_df[shared_df.price>600]
shared_expensive

long_island_df=airbnb_df[airbnb_df.neighbourhood=='Long Island City'].copy()
plt.hist(long_island_df.price, bins=np.arange(10,2500,50));

listings_df = airbnb_df.calculated_host_listings_count.sort_values(ascending=False)
listings_df.unique()
prolific_agents_df=airbnb_df[airbnb_df.calculated_host_listings_count>=96].copy()
prolific_agents_df
prolific_agents_df.shape
prolific_agents_df.id.unique()




import jovian
jovian.commit()
import jovian
jovian.commit()
import jovian
jovian.commit()

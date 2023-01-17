# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt

%matplotlib inline

import folium

from folium.plugins import FastMarkerCluster

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
listing_full = pd.read_csv('../input/cleansed_listings_dec18.csv',low_memory=False)

print('Number of listings: ', listing_full.shape[0])

print('Number of features in dataset: ', listing_full.shape[1])

print('Average price at: ${}'.format(round(listing_full.price.mean())))

print('Number of hosts: ', listing_full.host_id.nunique())

print('Number of suburbs: ',listing_full.zipcode.nunique())
useful_details = ['id','name','host_id','host_name', 'host_since', 'host_location', 'host_about',

                  'host_response_time', 'host_response_rate', 'host_is_superhost', 'host_neighborhood',

                  'host_verifications','host_identity_verified', 'city','suburb', 

                  'latitude', 'longitude','property_type', 

                  'room_type', 'accommodates','bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 

                  'price', 'weekly_price', 'monthly_price', 'security_deposit',

                  'cleaning_fee','minimum_nights','maximum_nights', 'calendar_updated', 

                  'has_availability','availability_30','availability_60', 'availability_90',

                  'availability_365', 'number_of_reviews','first_review','last_review',

                  'instant_bookable', 'cancellation_policy','calculated_host_listings_count', 

                  'reviews_per_month'

                 ] # the columns I selected

lis = listing_full[useful_details]

print(lis.columns.values)
# cleansing suburbs

kilda = lis[lis['suburb'].notnull()]

kilda = kilda[kilda['suburb'].str.contains('ilda')]

kilda['suburb'].unique()



lis['suburb'] = lis['suburb'].replace(['St. Kilda','Saint Kilda','St Kilda / Elwood', 'St kilda',

                                       'st kilda','Saint Kilda, Victoria, AU','Elwood, St. Kilda',

                                       'Saint Kilda Beach'],'St Kilda')



lis['suburb'] = lis['suburb'].replace(['Saint Kilda East','East St. Kilda','StKilda East',

                                       'Ripponlea (East St Kilda)','St.Kilda East','East st kilda',

                                       'St Kilda east'],'St Kilda East')

lis['suburb'] = lis['suburb'].replace(['Saint Kilda West','St Kilda West Melbourne','St. Kilda West'],

                                      'St Kilda West')





#plot top suburbs and inner cities based on the number of listings

fig, axarr = plt.subplots(1, 2, figsize=(16, 8))

subb = lis['suburb'].value_counts().sort_values(ascending=True).tail(10).plot.barh(

    ax=axarr[0], fontsize=12, color='Salmon',width=0.8)

axarr[0].set_title("Top 10 Suburbs", fontsize=20)

axarr[0].set_xlabel("Number of Listings", fontsize=12)



city = lis['city'].value_counts().sort_values(ascending=True).tail(10).plot.barh(

    ax=axarr[1],fontsize=12, color='IndianRed',width=0.8)

axarr[1].set_title("Top 10 Inner Cities", fontsize=20)

axarr[1].set_xlabel("Number of Listings", fontsize=12)
lat = lis['latitude']

lon = lis['longitude']

locations = list(zip(lat, lon))



map_mel = folium.Map(location=[-37.815018, 144.946014],tiles='CartoDB Positron',zoom_start=10 ) #'CartoDB dark_matter'  #melb location:-37.8136° N, 144.9631° E

FastMarkerCluster(data=locations).add_to(map_mel)

map_mel
rt = lis['room_type'].value_counts().sort_values(ascending=True)

rt.plot.barh(figsize=(15,2), linewidth = 1, width=0.8,color=['DimGrey','LightGray','Salmon'])

plt.title('Room Types in Melbourne', fontsize=20)

plt.xlabel('Number of listings', fontsize=12)
#plot property_type and room_type



#cleanse the property types

prop = lis.copy()

prop['property_type'] = prop['property_type'].replace(['Serviced apartment','Aparthotel','Condominium'],'Apartment')

prop['property_type'] = prop['property_type'].replace(['Townhouse','Guesthouse','Villa'],'House')



#only include top 5 property types

pt = []

for p in prop['property_type'].unique():

    if p not in prop.property_type.value_counts().sort_values(ascending=False).index[0:5]:

        pt.append(p)

    else:

        continue

        

prop['property_type'] = prop['property_type'].replace([pt],'other')



#plot

prop = prop.groupby(['property_type','room_type']).room_type.count().sort_values(ascending=False)

prop = prop.unstack()

prop['total'] = prop.iloc[:,0:3].sum(axis = 1)

prop = prop.sort_values(by=['total'])

prop = prop.drop(columns=['total'])



#plt.style.use('seaborn-white')

prop.plot.barh(stacked=True, color = ['Salmon','LightGray','DimGrey'],

              linewidth = 0.8, figsize=(15,4), width=0.8)

plt.title('Property and Room Types in Melbourne', fontsize=20)

plt.xlabel('Number of Listings', fontsize=12)

plt.ylabel('Property Types')

plt.legend(loc = 4,prop = {"size" : 12})
acc = lis['accommodates'].value_counts().sort_index(ascending=True)

acc.plot.bar(figsize=(15,4), color='pink', width=0.8)

plt.title("Accommodates (number of people)", fontsize=20)

plt.ylabel('Number of listings', fontsize=12)

plt.xlabel('Accommodates', fontsize=12)
# price by how many guests

pac = lis[lis['accommodates']<=6]

pac = pac.groupby('accommodates')['price'].mean().sort_values(ascending=True)

pac.plot.barh(figsize=(15, 4),grid=True, color='salmon', width=0.8)

plt.title("Average Daily Price by the Number of People a Property Accommodates", fontsize=20)

plt.xticks(np.arange(0, 250, step=40))

plt.xlabel('Average daily price (A$)', fontsize=12)

plt.ylabel("Number of people")
#price by property type and room type

prtype = lis.copy()

prtype['proroom'] = prtype['property_type']+prtype['room_type']



pr = prtype.proroom.value_counts()

prtype.proroom = prtype.proroom.map(pr)

prtype = prtype[prtype.proroom > 100] #only consider the combinatons of property and room types that have more 100 listings



ppr = prtype.groupby(['property_type', 'room_type'])['price'].mean().sort_values(ascending=True)

ppr.plot.barh(figsize=(15, 10),grid=True, color='pink', width=0.8)

plt.title("Average Daily Price by the Type of Property and Room", fontsize=20)

hn = max(prtype.groupby(['property_type', 'room_type'])['price'].mean())

plt.xticks(np.arange(0, hn, step=60))

plt.xlabel('Average Daily Price (A$)', fontsize=12)

plt.ylabel("Number of People")
#prop_type + room_type + accom

def plotPriceBySub(df, prop_type, room_type, accom, nlis, color, ttl):

    df2 = df[(df['property_type'] == prop_type)&(lis['room_type'] == room_type)&(lis['accommodates'] == accom)]

    sub = df2.suburb.value_counts()

    df2['total'] = df2['suburb']

    df2['total'] = df2['total'].map(sub)

    df2 = df2[df2['total']>= nlis] #only consider the combinations that have more than 20 listings

    df2 = df2.drop(columns=['total'])



    pri = df2.groupby('suburb')['price'].mean().sort_values(ascending=True).tail(10)

    pri.plot.barh(figsize=(20, 8), color=color, width=0.8)

    title = 'Average Daily Price for '+ ttl+' for '+ str(accom)+' People, by Suburbs'

    plt.title(title, fontsize=20)

    ph = max(df2.groupby('suburb')['price'].mean())

    plt.xticks(np.arange(0, ph,step=20))

    plt.xlabel('Average daily price (A$)', fontsize=12)

    plt.ylabel("Number of people")

    plt.show()

    

plotPriceBySub(lis, 'Apartment', 'Entire home/apt', 2, 20,'salmon','an Entire Apartment')

plotPriceBySub(lis, 'Apartment', 'Entire home/apt', 4, 20,'IndianRed','an Entire Apartment' )

plotPriceBySub(lis, 'Apartment', 'Private room', 2, 20,'LightSalmon','a Private Room in an Apartment')



plotPriceBySub(lis, 'House', 'Entire home/apt', 4, 10,'LightCoral','an Entire House')

plotPriceBySub(lis, 'House', 'Private room', 2, 20,'pink','a Private Room in a House')
hos = lis.groupby('host_id').size().reset_index(name='num_listings')

hos = hos.sort_values(by=['num_listings'],ascending=False)



print('{}% hosts have 1 listing.'.format(int(round(hos[hos['num_listings']==1].

                                                   host_id.count()*100/hos.host_id.count()))))

print('{}% hosts have 2 listing.'.format(int(round(hos[hos['num_listings']==2].

                                                   host_id.count()*100/hos.host_id.count()))))

print('{}% hosts have less than 5 listing.'.format(int(round(hos[hos['num_listings']<=5].

                                                             host_id.count()*100/hos.host_id.count()))))

print('{} hosts have more than 20 listings.'.format(hos[hos['num_listings']>=20].host_id.count()))

print('{} hosts have more than 50 listings.'.format(hos[hos['num_listings']>=50].host_id.count()))

print('The largest number of listings a host has is {}.'.format(max(hos['num_listings'])))
#make a copy

hosts = lis.copy()

hosts = hosts[['host_id','host_name','host_since','host_location','host_is_superhost',

                'host_identity_verified','host_verifications']]

#how many superhosts

print('{} Superhosts'.format(hosts.groupby('host_is_superhost').host_id.nunique()[1]))



# pro

prohos = lis[(lis['property_type']=='Serviced apartment')|

   (lis['property_type']=='Boutique hotel')|

   (lis['property_type']=='Hotel')|

   (lis['property_type']=='Aparthotel')|

   (lis['calculated_host_listings_count']>=10)].host_id.nunique()

print(prohos,'professional hosts.')



#cleanse host location

#hosts['host_location'] = hosts['host_location'].replace('AU','Australia')



#how many hosts are local and 

tothos = hosts['host_id'].nunique()

mehos = hosts[hosts['host_location'].str.contains('(?i)victoria',na=False)].host_id.nunique()

auhos = hosts[hosts['host_location'].str.contains('(?i)australia',na=False)].host_id.nunique()

cnhos = hosts[(hosts['host_location'].str.contains('(?i)china',na=False))|

             (hosts['host_location'].str.contains('(?i)cn',na=False))].host_id.nunique()

ushos = hosts[hosts['host_location'].str.contains('(?i)united states',na=False)].host_id.nunique()

ukhos = hosts[hosts['host_location'].str.contains('(?i)united kingdom',na=False)].host_id.nunique()

nzhos = hosts[hosts['host_location'].str.contains('(?i)zealand',na=False)].host_id.nunique()

nzhos



print('{} ({}%) hosts from Melbourne.'.format(mehos,int(round(mehos/tothos,2)*100)))

print('{} ({}%) hosts from Australia.'.format(auhos,int(round(auhos/tothos,2)*100)))

print('{} ({}%) hosts from China.'.format(cnhos, round(cnhos/tothos,4)*100))

print('{} ({}%) hosts from US.'.format(ushos,round(ushos*100/tothos,2)))

print('{} ({}%) hosts from UK.'.format(ukhos,round(ukhos*100/tothos,2)))

print('{} ({}%) hosts from New Zealand.'.format(nzhos,round(nzhos*100/tothos,2)))
small_details = ['id','name','host_id', 'city','suburb', 'property_type', 

                  'room_type', 'accommodates','bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 

                  'price', 'weekly_price', 'monthly_price', 'security_deposit',

                  'cleaning_fee','minimum_nights','maximum_nights', 

                  'instant_bookable', 'cancellation_policy','calculated_host_listings_count', 

                  'reviews_per_month'

                 ] 

lismal = lis[small_details]



# convert property_type 

lismal['property_type'] = lismal['property_type'].replace(['Hotel','Farm stay',

        'Cottage', 'Other', 'Boutique hotel',

       'Earth house', 'Bungalow', 'Tiny house', 'Nature lodge',

       'Cabin', 'Hostel', 'Barn', 'Train', 'Boat', 'Camper/RV',

       'Campsite', 'Treehouse', 'Tent', 'Chalet', 'Aparthotel', 'Castle',

       'Resort', 'Hut', 'Minsu (Taiwan)', 'Casa particular (Cuba)'],'Rare')

proty_mapping = {'Apartment':10, 'House ':9, 'Townhouse':8, 'Condominium':7, 'Serviced apartment':6,'Villa':5,

                'Guesthouse':4, 'Bed and breakfast':3,'Guest suite':2, 'Loft':1, 'Rare':0}



lismal['property_type'] = lismal['property_type'].map(proty_mapping)



#convert room_type

roty_mapping = {'Entire home/apt':3, 'Private room':2, 'Shared room':1}



lismal['room_type'] = lismal['room_type'].map(roty_mapping)



# plot heatmap

corr = lismal.corr()

plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

corrMat = plt.matshow(corr, fignum = 1)

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.gca().xaxis.tick_bottom()

plt.colorbar(corrMat)

plt.title('Correlation Matrix for Listings features', fontsize=15)

plt.show()
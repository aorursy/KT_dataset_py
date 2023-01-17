import pandas as pd

import math

import numpy as bp

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
data.shape
data.head(3)
data.describe()
data.info()
data['hotel'].value_counts()
#The 

room_type = list(data['reserved_room_type'].value_counts().index)

room_type_num = list(data['reserved_room_type'].value_counts())

len(room_type)

len(room_type_num)

sns.barplot(x = room_type, y = room_type_num)

plt.title('The popularity of different room types')

plt.xlabel('Room type')

plt.ylabel('Frequency')
ch = data[(data['hotel'] == 'City Hotel') & (data['is_canceled'] == 0)]

rh = data[(data['hotel'] == 'Resort Hotel') & (data['is_canceled'] == 0)]



ch['total_nights'] = ch['stays_in_weekend_nights'] + ch['stays_in_week_nights']

rh['total_nights'] = rh['stays_in_weekend_nights'] + rh['stays_in_week_nights']

data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
ch['adr_pp'] = ch['adr']/(ch['adults'] + ch['children'])

rh['adr_pp'] = rh['adr']/(rh['adults'] + rh['children'])
data['adr_pp'] = data['adr']/(data['adults'] + data['children'])

all_number = data[data['is_canceled'] == 0]

room_prices = all_number[['hotel', 'reserved_room_type', 'adr_pp']].sort_values('reserved_room_type')
sns.barplot(x = 'reserved_room_type', y = 'adr_pp', hue = 'hotel', data = room_prices, hue_order = ['City Hotel', 'Resort Hotel'], ci = 'sd')

plt.title('Price of room types per night and person')

plt.xlabel('Room type')

plt.ylabel('Price')

plt.legend(loc = 1)

plt.show()
corr = data.corr()

sns.heatmap(corr)
plt.scatter(data['previous_bookings_not_canceled'], data['is_repeated_guest'])

plt.xlabel("previous_bookings_not_canceled")

plt.ylabel("is_repeated_guest")

plt.title("Relationship")

plt.show()
plt.scatter(data['previous_cancellations'], data['is_repeated_guest'])

plt.xlabel("previous_cancellations")

plt.ylabel("is_repeated_guest")

plt.title("Relationship")

plt.show()
col = data.columns

for i in col:

    count = data[i].value_counts()

    print(count)


adults_count = data['adults'].value_counts()

adults_count

ad = {'2':89680,'1':23027,'3':6202,'0':403,'4':62,'26':5,'27':2,'20':2,'5':2,'55':1,'50':1,'40':1,'10':1,'6':1}

ad_names = list(ad.keys())

ad_values = list(ad.values())

plt.bar(ad_names,ad_values,color='red')

plt.ylabel("total number of adults")

plt.xlabel("types of adults")
plt.hist(data['adults'])

plt.xlabel('adults')

plt.ylabel('frequency')

plt.title('histofram of adults')

plt.show()
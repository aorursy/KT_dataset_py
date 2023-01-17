# Import libraries and get data into memory for exploration

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



app_labels = pd.read_csv('../input/app_labels.csv')

label_categories = pd.read_csv('../input/label_categories.csv')

app_events = pd.read_csv('../input/app_events.csv')

events = pd.read_csv('../input/events.csv')

phone_brand_device_model = pd.read_csv('../input/phone_brand_device_model.csv')

gender_age_train = pd.read_csv('../input/gender_age_train.csv')

gender_age_test = pd.read_csv('../input/gender_age_test.csv')
def print_data_summary(data, name):

    print("\n******** %s ********\n%s" % (name, data.head()))

    print("\n%s.shape: %s" % (name, data.shape))



    num_unique_items = data.apply(lambda x: x.nunique())

    print("\nNumber of unique items in each column:\n%s" % (num_unique_items,))
print_data_summary(app_labels, 'app_labels')

print_data_summary(label_categories, 'label_categories')
apps_labels_cats = pd.merge(app_labels, label_categories, how='left')



print("Following are the top 25 labels that occur most frequently.\n\nCategory\t\tNumber of occurances")

print(apps_labels_cats.category.value_counts()[:25])



print("\nFollowing is distribution of label-counts.\nCount Number of apps")

print(app_labels.app_id.value_counts().value_counts())
print_data_summary(events, 'events')
from datetime import datetime

from datetime import timedelta



# Convert dates from string to datetime objects

events.timestamp = pd.to_datetime(events.timestamp)

print ("Min Time: %s\tMax Time: %s" % (events.timestamp.min(), events.timestamp.max()))

print ("Perentage of events between 1st May and 8th May: %.2f %%" % 

       (np.sum((events.timestamp > datetime(2016, 5, 1)) & (events.timestamp < datetime(2016, 5, 8)))/float(len(events))*100))
events['day'] = events.timestamp.apply(lambda x: x.day)

day_distrib = events.day.value_counts().sort_index()



events['hour'] = events.timestamp.apply(lambda x: x.hour)

hourly_distrib = events.hour.value_counts().sort_index()
fig, axes = plt.subplots(1, 2)

fig.set_size_inches(12, 4)



day_distrib.plot.bar(ax=axes[0])

axes[0].set_xticklabels(['5 / 1', '5 / 2', '5 / 3', '5 / 4', '5 / 5', '5 / 6', '5 / 7', '5 / 8', '4 / 30'])

axes[0].set_xlabel('Day')

axes[0].set_ylabel('Events Count')

axes[0].set_title('Day-wise Distribution')



hourly_distrib.plot.bar(ax=axes[1])

axes[1].set_xlabel('Hour')

axes[1].set_ylabel('Events Count')

axes[1].set_title('Hourly Distribution')
days = np.unique(events.day)

day_hour_distrib = events.groupby(['day']).apply(lambda x: x.hour.value_counts())

day_hour_distrib = day_hour_distrib.unstack(level=0)



fig, axes = plt.subplots(4,2)

fig.set_size_inches(12,20)

for i, col in enumerate(day_hour_distrib.columns[:7]):

    ax = axes[i/2, i%2]

    ax.bar(range(24), day_hour_distrib[col])

    ax.set_xticklabels(range(24))

    ax.set_xlabel('hour')

    ax.set_ylabel('count')

    ax.set_title('May-'+str(col))
gender_age_events = events.merge(gender_age_train, how='left', left_on='device_id', right_on='device_id')

day_gender_distrib = gender_age_events.groupby('gender')['day'].value_counts().unstack(level=0)

hour_gender_distrib = gender_age_events.groupby('gender')['hour'].value_counts().unstack(level=0)
fig, axes = plt.subplots(2, 2)

fig.set_size_inches(10, 10)



ax = axes[0, 0]

day_gender_distrib['M'].plot.bar(ax=ax)

ax.set_xticklabels(['5 / 1', '5 / 2', '5 / 3', '5 / 4', '5 / 5', '5 / 6', '5 / 7', '5 / 8', '4 / 30'])

ax.set_xlabel('Day')

ax.set_ylabel('Count')

ax.set_title('Day-wise Event Distribution - Male')



ax=axes[0, 1]

day_gender_distrib['F'].plot.bar(ax=ax)

ax.set_xticklabels(['5 / 1', '5 / 2', '5 / 3', '5 / 4', '5 / 5', '5 / 6', '5 / 7', '5 / 8', '4 / 30'])

ax.set_xlabel('Day')

ax.set_ylabel('Count')

ax.set_title('Day-wise Event Distribution - Female')



ax=axes[1, 0]

hour_gender_distrib['M'].plot.bar(ax=ax)

ax.set_xlabel('Hour')

ax.set_ylabel('Count')

ax.set_title('Hourly Event Distribution - Male')



ax=axes[1, 1]

hour_gender_distrib['F'].plot.bar(ax=ax)

ax.set_xlabel('Hour')

ax.set_ylabel('Count')

ax.set_title('Hourly Event Distribution - Female')



plt.tight_layout()
print("***** Latitude data *****\n\nLatitude Count")

print(events.latitude.value_counts().head())



print("\nlatitude = 0: %.2f%%" % (np.sum(events.latitude == 0)/float(len(events)) * 100))

print("latitude = 1: %.2f%%" % (np.sum(events.latitude == 1)/float(len(events)) * 100))

print("20 < latitude < 55: %.2f%%" % (np.sum((events.latitude > 20) & (events.latitude < 55))/float(len(events)) * 100))



print("\n***** Longitude data *****\n\nLongitude Count")

print(events.latitude.value_counts().head())



print("\nlongitude = 0: %.2f%%" % (np.sum(events.longitude == 0)/float(len(events)) * 100))

print("longitude = 1: %.2f%%" % (np.sum(events.longitude == 1)/float(len(events)) * 100))

print("75 < longitude < 135: %.2f%%" % (np.sum((events.longitude > 75) & (events.longitude < 135))/float(len(events)) * 100))
print_data_summary(app_events, 'app_events')



print("\nis_active distribution (in percentage):")

print(app_events.is_active.value_counts() / float(len(app_events)) * 100)
eventsid_distrib = app_events.event_id.value_counts()

print('Occurances event_id-Count')

print(eventsid_distrib.value_counts().sort_index().head(10))

eventsid_distrib.hist(bins=eventsid_distrib.max())

plt.xlabel('Frequency')

plt.ylabel('Count')

plt.title('event_id Distribution')
appid_distrib = app_events.app_id.value_counts()

print('app_id\t\t\tRecord-Count')

print(appid_distrib.head(10))

print('\nOccurances app_id-Count')

print(appid_distrib.value_counts().sort_index().head(10))



plt.bar(range(len(appid_distrib)), appid_distrib)

plt.xlabel('Frequency')

plt.ylabel('Count')

plt.title('app_id Distribution')
print_data_summary(phone_brand_device_model, 'phone_brand_device_model')
#english_phone_brand_device_model = pd.read_csv('../Data/English_phone_brand_device_model.csv')

print("Top 10 phone brands:\nBrand\tDevice Count")

brand_market_share = phone_brand_device_model.phone_brand.value_counts()

print(brand_market_share.head(10))
plt.figure().set_size_inches(12,8)

brand_market_share.plot.bar()

#plt.bar(range(len(brand_market_share)), brand_market_share)

plt.xticks(range(len(brand_market_share)), brand_market_share.index)

plt.xlabel('Phone Brands')

plt.ylabel('Device Count')

plt.title('Device counts of Phone Brands')
phone_brand_device_model['brand_model'] = phone_brand_device_model.phone_brand.str.cat(phone_brand_device_model.device_model, sep='_')

model_cnt = phone_brand_device_model.groupby('brand_model')['device_model'].count()

pop_model_cnt = model_cnt[model_cnt > 500]

#plt.bar(range(len(pop_model_cnt)), pop_model_cnt)

plt.figure().set_size_inches(12, 10)

pop_model_cnt.plot.bar()

plt.title('Popular Device Models Distribution')

plt.xlabel('Brand_Model Names')

plt.ylabel('Device Count')
model_cnts = phone_brand_device_model.groupby('phone_brand').device_model.unique()

model_cnts = model_cnts.apply(lambda x: len(x))

plt.figure().set_size_inches(12, 8)

model_cnts.plot.bar()

plt.title('Brand-wise Models Counts')

plt.ylabel('Brand Count')
print_data_summary(gender_age_train, 'gender_age_train')

print_data_summary(gender_age_test, 'gender_age_test')

train_pct = gender_age_train.shape[0] / float(len(gender_age_train) + len(gender_age_test))

print('\n=========== train:test split = %.2f : %.2f ==============' % (train_pct, 1.0-train_pct))
plt.figure(figsize=(12,5))

gender_age_train.age.hist(bins=gender_age_train.age.max())

plt.xticks(range(1, gender_age_train.age.max()+1))

plt.xlabel('age')

plt.ylabel('count')

plt.title('Age Distribution')
from sklearn.preprocessing import LabelEncoder



agegroup_enc = LabelEncoder()

gender_age_train.group = agegroup_enc.fit_transform(gender_age_train.group)

group_distrib = gender_age_train.group.value_counts().sort_index()
plt.figure(figsize=(8,5))

plt.bar(group_distrib.index, group_distrib)

plt.xticks(group_distrib.index, agegroup_enc.classes_)

plt.xlabel('Age Groups')

plt.ylabel('Count')

plt.title('Age Group Distribution')
gender_age_train.gender.value_counts() / float(len(gender_age_train))
brand_model_gender_age_groups = gender_age_train.merge(phone_brand_device_model, how='left')

gender_wise_brand_distrib = brand_model_gender_age_groups.groupby('gender').phone_brand.value_counts().unstack(level=0)



fig, axes = plt.subplots(2,1)

fig.set_size_inches(10,12)

ax = axes[0]

gender_wise_brand_distrib['M'].plot.bar(ax=ax)

ax.set_xlabel('Phone Brands')

ax.set_ylabel('Device Count')

ax.set_title('Brand Distribution - Male')

ax = axes[1]

gender_wise_brand_distrib['F'].plot.bar(ax=ax)

ax.set_xlabel('Phone Brands')

ax.set_ylabel('Device Count')

ax.set_title('Brand Distribution - Female')

fig.tight_layout()
group_wise_brand_distrib = brand_model_gender_age_groups.groupby('group').phone_brand.value_counts().unstack(level=0)



groups = group_wise_brand_distrib.columns

grp_names = agegroup_enc.classes_

brands = gender_wise_brand_distrib.M[gender_wise_brand_distrib.M > 500].index

group_wise_brand_distrib_top_brands = group_wise_brand_distrib.ix[brands]

num_groups = len(groups)

fig, axes = plt.subplots(6,2)

fig.set_size_inches(10, 4*6)

for i in range(int(num_groups/2)):

    ax = axes[i,0]

    group_wise_brand_distrib_top_brands[groups[i]].plot.bar(ax=ax)

    ax.set_xlabel('Phone Brands')

    ax.set_ylabel('Device Count')

    ax.set_title('Top Brands Distribution - %s'%(grp_names[i]))



    ax = axes[i,1]

    group_wise_brand_distrib_top_brands[groups[i+6]].plot.bar(ax=ax)

    ax.set_xlabel('Phone Brands')

    ax.set_ylabel('Device Count')

    ax.set_title('Top Brands Distribution - %s'%(grp_names[i+6]))



fig.tight_layout()
brand_model_gender_age_groups['brand_model'] = brand_model_gender_age_groups.phone_brand.str.cat(brand_model_gender_age_groups.device_model, sep='_')

gender_wise_brand_model_distrib = brand_model_gender_age_groups.groupby(['brand_model', 'gender']).device_model.count().unstack()



f = gender_wise_brand_model_distrib.F[gender_wise_brand_model_distrib.F > 50]

m = gender_wise_brand_model_distrib.M[gender_wise_brand_model_distrib.M > 100]



fig, axes = plt.subplots(2, 1)

fig.set_size_inches(10, 12)

ax = axes[0]

f.plot.bar(ax=ax)

ax.set_xlabel('Brand_Model name')

ax.set_ylabel('Device Count')

ax.set_title('Popular Brand_Model Distribution - Female')

ax = axes[1]

m.plot.bar(ax=ax)

ax.set_xlabel('Brand_Model name')

ax.set_ylabel('Device Count')

ax.set_title('Popular Brand_Model Distribution - Male')



fig.tight_layout()
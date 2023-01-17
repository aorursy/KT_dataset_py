import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.tseries.offsets import BDay

import calendar

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

chicago_crimes_df = pd.read_csv('../input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv')

# relevant columns

crime_features = chicago_crimes_df[['Date','Location', 'District', 'Primary Type']]

# relevant crime types  top ten most frequent

top_10_count=pd.DataFrame(chicago_crimes_df['Primary Type'].value_counts().head(10))

# type names of top ten most frequent

top_ten_types = top_10_count.index

# filter relevant rows

crime_features = crime_features[crime_features['Primary Type'].map(lambda x: x in top_ten_types)]

# drop rows containing missing values

crime_features.dropna(inplace=True)

# convert date strings to datetime index

crime_features.Date = pd.to_datetime(crime_features.Date, format='%m/%d/%Y %I:%M:%S %p')

# set as index

crime_features.index = crime_features.Date 

# sort rows by date

crime_features.sort_index(inplace=True)

# change column name for convenience

crime_features.rename(columns={'Primary Type': 'Type'}, inplace=True)

crime_features.head()
# convert datetime to feature columns

crime_features['dayofweek'] = crime_features.index.weekday

crime_features['month'] = crime_features.index.month

crime_features['week'] = crime_features.index.week

crime_features['hour'] = crime_features.index.hour

crime_features['year'] = crime_features.index.year



# business-day feature

isBusinessDay = BDay().onOffset

match_series = pd.to_datetime(crime_features.index).map(isBusinessDay).astype(int)

crime_features['workday'] = match_series



# eight hour interval feature

crime_features['hour_interval'] = crime_features.index.floor('8H').hour

crime_features.head()
for c_type in top_ten_types:

    

    fig = plt.figure(figsize=(15,10))

    fig.suptitle(c_type, fontsize='x-large')

    a=fig.add_subplot(1,2,1)

    max_year = crime_features[crime_features['Type']==c_type]['year'].value_counts().index[0] 

    sns.countplot('year',data=crime_features[crime_features['Type']==c_type],

                  order=range(2012,2017),palette='dark')

    plt.xlabel("Year", fontsize=14)

    plt.ylabel("No. of Crimes Reported", fontsize=14)

    plt.title("{}: crimes reported by year \n with maximum at year {} ".format(

                    c_type, max_year),fontsize=12)

    

    a=fig.add_subplot(1,2,2)

    max_month = crime_features[crime_features['Type']==c_type]['month'].value_counts().index[0]

    max_month_name = calendar.month_name[max_month]

    sns.countplot('month',data=crime_features[crime_features['Type']==c_type],

                  order=range(1,13),palette="Reds_d")

    plt.xlabel("Month", fontsize=14)

    plt.ylabel("No. of Crimes Reported", fontsize=14)

    plt.title("{}: crimes reported by month \n with maximum at {} ".format(

                c_type, max_month_name),fontsize=12)



    plt.show()



crime_features.District.unique().astype(int)
district_nums = crime_features.District.unique().astype(int)

for c_type in top_ten_types:

    fig = plt.figure(figsize=(15,10))

    fig.suptitle(c_type, fontsize='x-large')

    a=fig.add_subplot(1,2,1)

    max_day = crime_features[crime_features['Type']==c_type]['dayofweek'].value_counts().index[0]

    max_day_name = calendar.day_name[max_day]

    sns.countplot('dayofweek',data=crime_features[crime_features['Type']==c_type],

                      order=range(7),palette="Reds_d")

    plt.xlabel("Day", fontsize=14)

    plt.ylabel("No. of crimes reported", fontsize=14)

    plt.title("{}: crimes reported per day \n with maximum at {} ".format(

                c_type, max_day_name),fontsize=12)

    

    a=fig.add_subplot(1,2,2)

    max_district  = crime_features[crime_features['Type']==c_type]['District'].value_counts().index[0]

    sns.countplot('District',data=crime_features[crime_features['Type']==c_type], 

                      order=district_nums, palette="Greens_d")



    plt.xlabel("District", fontsize=14)

    plt.ylabel("No. of crimes reported", fontsize=14)

    plt.title("{}: crimes reported by district \n with maximum at district {} ".format(

                c_type, int(max_district)),fontsize=12)



    plt.show()
for c_type in top_ten_types:

    fig = plt.figure(figsize=(15,10))

    fig.suptitle(c_type, fontsize='x-large')

    a=fig.add_subplot(1,2,1)

    max_hour = crime_features[crime_features['Type']==c_type]['hour'].value_counts().index[0]

    sns.countplot('hour',data=crime_features[crime_features['Type']==c_type],

                      order=range(24),palette="Greens_d")



    plt.xlabel("Hour", fontsize=14)

    plt.ylabel("No. of crimes reported", fontsize=14)

    plt.title("{}: crimes reported by hour \n with maximum at {} ".format(

                c_type, max_hour),fontsize=12)

    a=fig.add_subplot(1,2,2)

    max_hour_interval  = crime_features[crime_features['Type']==c_type]['hour_interval'].value_counts().index[0]

    sns.countplot('hour_interval',data=crime_features[crime_features['Type']==c_type])



    plt.xlabel("Eight Hour Inteval", fontsize=14)

    plt.ylabel("No. of crimes reported", fontsize=14)

    plt.title("{}: crimes reported by eight hour intervals \n with maximum at {} to {}".format(

        c_type, max_hour_interval, max_hour_interval+8),fontsize=12)



    plt.show()
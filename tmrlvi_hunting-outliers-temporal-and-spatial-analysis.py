%pylab inline

import os

import pandas as pd

import seaborn as sns

from haversine import AVG_EARTH_RADIUS
train = pd.read_csv("../input/train.csv", index_col=0)
train = train.assign(

    pickup_datetime = pd.to_datetime(train.pickup_datetime),

    dropoff_datetime = pd.to_datetime(train.dropoff_datetime),

    vendor_id = train.vendor_id.astype("category"),

    store_and_fwd_flag = train.store_and_fwd_flag.astype("category"),

    passenger_count = train.passenger_count.astype("category", ordered=True),

    dlat = (train.pickup_latitude - train.dropoff_latitude) * pi/180,

    dlong = (train.pickup_longitude - train.dropoff_longitude) * pi/180

).assign(

    euclidean_distance = lambda df: (2* AVG_EARTH_RADIUS*

                                     arcsin(sqrt(square(sin(df.dlat/2)) + 

                                            cos(df.pickup_latitude * pi/180) * 

                                            cos(df.dropoff_latitude * pi/180) * 

                                            square(sin(df.dlong/2)))))

)
train = train[

    (train.trip_duration < 500000) &

    (train.euclidean_distance < 125)

].copy()
def summary(df, sample_size = None):

    if sample_size is None or sample_size > len(df):

        sample_size = len(df)

    sample = np.random.choice(df.index, size=sample_size, replace=False)

    fig, axes = plt.subplots(3,3, figsize=(12,12))

    axes = np.reshape(axes, -1)

    for index, field in enumerate(['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',

                                   'dropoff_longitude', 'dropoff_latitude', 'trip_duration', 'store_and_fwd_flag',

                                   'euclidean_distance']):

        data = df.loc[sample, field]

        if isinstance(data.dtype, pd.core.dtypes.dtypes.CategoricalDtype):

            sns.countplot(x=field, data=df.loc[sample, [field]], ax=axes[index])

        else:

            sns.violinplot(data=df.loc[sample, [field]], ax=axes[index])

        axes[index].set_title(field)



summary(train, 10000)
fig, axes = subplots(1, 2, figsize=(15,5))

train.euclidean_distance.hist(bins=100, ax=axes[0])

axes[0].set_yscale("log")

axes[0].set_title("Distance Historgram")



train.trip_duration.hist(bins=100, ax=axes[1])

axes[1].set_yscale("log")

_ = axes[1].set_title("Trip Duration Historgram")
DAY_NAME = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]



def weekday_hour_table(df, func):

    """

    df - the dataframe to summaries

    func - function for calculating the values for each pair of pickup-dropoff times

    """

    df = df.groupby(

        [

            df.pickup_datetime.apply(lambda dt: (dt.weekday(), dt.hour)),

            df.dropoff_datetime.apply(lambda dt: (dt.weekday(), dt.hour)),

        ]

    ).apply(func).unstack()

    df.index = ["%s %d" % (DAY_NAME[day], hour) for day, hour in df.index]

    df.columns = ["%s %d" % (DAY_NAME[day], hour) for day, hour in df.columns]

    return df



def median_of_average_ride_speed(group):

    return (group.euclidean_distance / group.trip_duration).median()



traffic_by_day = weekday_hour_table(train, median_of_average_ride_speed)

figsize(15,15)

ax = sns.heatmap(log(traffic_by_day+1e-7), xticklabels=2, yticklabels=2)

_= xticks(rotation=90)
figsize(15,5)

df = train[

    (train.dropoff_datetime.dt.hour == 0) &

    (train.dropoff_datetime.dt.minute == 0) &

    (train.dropoff_datetime.dt.second == 0)

]

sample = np.random.choice(df.index, size=min(10000, len(df)), replace=False)

_ = sns.stripplot(y="trip_duration", x="vendor_id", data=df.loc[sample].copy(), jitter=True)
traffic_by_day = weekday_hour_table(train[train.vendor_id == 1], median_of_average_ride_speed)

figsize(15,10)

ax = sns.heatmap(log(traffic_by_day+1e-7), xticklabels=2, yticklabels=2)

_= xticks(rotation=90)
outliers = (

    ((

        (train.vendor_id == 2) &

        (train.dropoff_datetime.dt.hour == 0) &

        (train.dropoff_datetime.dt.minute == 0) &

        (train.dropoff_datetime.dt.second == 0)

    ) |

    (

        train.trip_duration > 40000

    ) | 

    (

        train.euclidean_distance == 0

    ))

)

train = train[~outliers].copy()
fig, axes = subplots(1, 2, figsize=(15,5))

train.euclidean_distance.hist(bins=100, ax=axes[0])

(train["euclidean_distance"] / (train["trip_duration"]/3600)).hist(bins=200, ax=axes[0])

axes[0].set_yscale("log")

axes[0].set_title("KM Per Hour Historgram")



((train["trip_duration"]/3600) / train["euclidean_distance"]).hist(bins=200, ax=axes[1])

axes[1].set_yscale("log")

_ = axes[1].set_title("Hour Per KM Historgram")
too_fast = ((train["euclidean_distance"] / (train["trip_duration"]/3600)) > 300)

too_slow = ((train["euclidean_distance"] / (train["trip_duration"]/3600)) < 0.5)

# Removing some outliers to make the plot clearer

not_too_far = (train["pickup_latitude"] > 40) &  (train["pickup_longitude"] > -76)



#Plotting

with_speed = train[(too_fast | too_slow) & not_too_far].copy()

with_speed["fast"] = too_fast

_ = sns.lmplot(x='pickup_longitude', y='pickup_latitude', hue="fast", markers='.', size=10, 

               fit_reg=False, data=with_speed)
train = train[~too_fast & ~too_slow].copy()
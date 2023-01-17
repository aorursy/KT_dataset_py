# import necessary package

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score





%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
# load data

seattle_calendar = pd.read_csv('../input/calendar.csv')

seattle_listing = pd.read_csv('../input/listings.csv')

seattle_review = pd.read_csv('../input/reviews.csv')
seattle_calendar.head()
seattle_calendar.info()
#  If the available values are f, the price values seems to be NaN. But it is only a hypothesis, is it true all data?

calendar_q1_df = seattle_calendar.groupby('available')['price'].count().reset_index()

calendar_q1_df.columns = ['available', 'price_nonnull_count']

calendar_q1_df
#  How many rows per each listing_id?

calendar_q2_df = seattle_calendar.groupby('listing_id')['date'].count().reset_index()

calendar_q2_df['date'].value_counts()
# process data

calendar_q3_df = seattle_calendar.copy(deep=True)

calendar_q3_df.dropna(inplace=True)

calendar_q3_df['date'] = pd.to_datetime(calendar_q3_df['date'])

calendar_q3_df['price'] = calendar_q3_df['price'].map(lambda x: float(x[1:].replace(",", "")))



# apply aggregation

calendar_q3_df = calendar_q3_df.groupby('date')['price'].mean().reset_index()



# plot avg listings prices over time.

plt.figure(figsize=(15, 8))

plt.plot(calendar_q3_df.date, calendar_q3_df.price, color='b', marker='.', linewidth=0.9)

plt.title("Average listing price by date")

plt.xlabel('date')

plt.ylabel('average listing price')

plt.grid()
# plot more narrow range

plt.figure(figsize=(15, 8))

plt.plot(calendar_q3_df.date.values[:15], calendar_q3_df.price.values[:15], color='b', marker='o', linewidth=1.5)

plt.title("Average listing price by date")

plt.xlabel('date')

plt.ylabel('average listing price')

plt.grid()
# create weekday column

calendar_q3_df["weekday"] = calendar_q3_df["date"].dt.weekday_name



# boxplot to see price distribution

plt.figure(figsize=(15, 8))

sns.boxplot(x = 'weekday',  y = 'price', data = calendar_q3_df, palette="Blues", width=0.6)

plt.show()
seattle_listing.head()
print(list(seattle_listing.columns.values))
print("Num of listings: ", seattle_listing.id.count())

print("Num of rows: ", seattle_listing.shape[0])
seattle_listing['review_scores_rating'].describe().reset_index()
# cleaning data

listings_q1_df = seattle_listing['review_scores_rating'].dropna()



# plot histgram

plt.figure(figsize=(15, 8))

plt.hist(listings_q1_df.values, bins=80, color='b')

plt.grid()
# cleaning data

listings_q2_df = seattle_listing.copy(deep=True)

listings_q2_df = listings_q2_df['price'].dropna().reset_index()

listings_q2_df['price'] = listings_q2_df['price'].map(lambda x: float(x[1:].replace(',', '')))



listings_q2_df['price'].describe().reset_index()
plt.figure(figsize=(15, 8))

plt.hist(listings_q2_df.price, bins=100, color='b')

plt.grid()
seattle_listing['maximum_nights'].describe().reset_index()
# eliminate outliers because maximum values are very large.

listings_q3_df = seattle_listing[seattle_listing['maximum_nights'] <= 1500]



plt.figure(figsize=(15, 8))

plt.hist(listings_q3_df.maximum_nights, bins=100, color='b')

plt.xlabel('maximum nights')

plt.ylabel('listings count')

plt.grid()
seattle_review.head()
seattle_review.info()
print("sample 1: ", seattle_review.comments.values[0], "\n")

print("sample 2: ", seattle_review.comments.values[3])
# convert date column's data type to date from object

review_q1_df = seattle_review.copy(deep=True)

review_q1_df.date = pd.to_datetime(review_q1_df.date)



review_q1_df = review_q1_df.groupby('date')['id'].count().reset_index()



# plot avg listings prices over time.

plt.figure(figsize=(15, 8))

plt.plot(review_q1_df.date, review_q1_df.id, color='b', linewidth=0.9)

plt.title("Number of reviews by date")

plt.xlabel('date')

plt.ylabel('number of reviews')

plt.grid()
# create rolling mean column

review_q1_df["rolling_mean_30"] = review_q1_df.id.rolling(window=30).mean()



# plot avg listings prices over time.

plt.figure(figsize=(15, 8))

plt.plot(review_q1_df.date, review_q1_df.rolling_mean_30, color='b', linewidth=2.0)

plt.title("Number of reviews by date")

plt.xlabel('date')

plt.ylabel('number of reviews')

plt.grid()
review_q1_df["year"] = review_q1_df.date.dt.year

years = review_q1_df.year.unique()



for year in years:

    if year >= 2010 and year < 2016:

        year_df = review_q1_df[review_q1_df.year == year]

        max_value = year_df.rolling_mean_30.max()

        max_date = year_df[year_df.rolling_mean_30 == max_value].date.dt.date.values[0]

        print(year, max_date, np.round(max_value, 1))
listings_q3_df["min_max_night_diff"] = listings_q3_df.maximum_nights - listings_q3_df.minimum_nights



plt.figure(figsize=(15, 8))

plt.plot(listings_q3_df.maximum_nights, listings_q3_df.minimum_nights, color='b', marker='o', linewidth=0, alpha=0.25)

plt.xlabel('maximum nights')

plt.ylabel('minimum nights')

plt.grid()
review_q2_df = review_q1_df[review_q1_df.year == 2015]



plt.figure(figsize=(15, 8))

plt.plot(review_q2_df.date, review_q2_df.rolling_mean_30, color='b', linewidth=2.0)

plt.title("Number of reviews by date")

plt.grid()
prepare_df = seattle_listing.copy(deep=True)
# check null count

df_length = prepare_df.shape[0]



for col in prepare_df.columns:

    null_count = prepare_df[col].isnull().sum()

    if null_count == 0:

        continue

        

    null_ratio = np.round(null_count/df_length * 100, 2)

    print("{} has {} null values ({}%)".format(col, null_count, null_ratio))
# detect need drop columns

drop_cols = [col for col in prepare_df.columns if prepare_df[col].isnull().sum()/df_length >= 0.3]



# drop null

prepare_df.drop(drop_cols, axis=1, inplace=True)

prepare_df.dropna(subset=['host_since'], inplace=True)



# check after

for col in prepare_df.columns:

    null_count = prepare_df[col].isnull().sum()

    if null_count == 0:

        continue

        

    null_ratio = np.round(null_count/df_length * 100, 2)

    print("{} has {} null values ({}%)".format(col, null_count, null_ratio))
drop_cols = ['listing_url', 'scrape_id', 'last_scraped', 'name', 'summary', 'space', 'description', 'neighborhood_overview',

                'transit', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url', 'host_name', 'host_about', 'host_thumbnail_url',

                'host_picture_url', 'street', 'city', 'state', 'zipcode', 'market', 'smart_location', 'country_code', 'country', 'latitude', 'longitude',

                'calendar_updated', 'calendar_last_scraped', 'first_review', 'last_review', 'amenities', 'host_verifications']



prepare_df.drop(drop_cols, axis=1, inplace=True)
prepare_df.columns
drop_cols = []

for col in prepare_df.columns:

    if prepare_df[col].nunique() == 1:

        drop_cols.append(col)

        

prepare_df.drop(drop_cols, axis=1, inplace=True)

prepare_df.columns
# available days count each listings

listing_avalilable = seattle_calendar.groupby('listing_id')['price'].count().reset_index()

listing_avalilable.columns = ["id", "available_count"]



# merge

prepare_df = prepare_df.merge(listing_avalilable, how='left', on='id')



# create target column

prepare_df['host_since_year'] = pd.to_datetime(prepare_df['host_since']).dt.year

prepare_df["easily_accomodated"] = prepare_df.accommodates / (prepare_df.available_count+1) / (2017 - prepare_df.host_since_year)
print("Before: {} columns".format(prepare_df.shape[1]))



drop_cols = ['host_since', 'accommodates', 'availability_30', 'availability_60', 'availability_90', 'availability_365',

                'number_of_reviews', 'review_scores_rating', 'available_count', 'reviews_per_month', 'host_since_year', 'review_scores_value']



prepare_df.drop(drop_cols, axis=1, inplace=True)

print("After: {} columns".format(prepare_df.shape[1]))
# convert true or false value to 1 or 0

dummy_cols = ['host_is_superhost', 'require_guest_phone_verification', 'require_guest_profile_picture', 'instant_bookable', 

              'host_has_profile_pic', 'host_identity_verified', 'is_location_exact']



for col in dummy_cols:

    prepare_df[col] = prepare_df[col].map(lambda x: 1 if x == 't' else 0)



# create dummy valuables

dummy_cols = ['host_location', 'host_neighbourhood', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',

             'property_type', 'room_type', 'bed_type', 'cancellation_policy', 'host_response_time']



prepare_df = pd.get_dummies(prepare_df, columns=dummy_cols, dummy_na=True)
df_length = prepare_df.shape[0]



for col in prepare_df.columns:

    null_count = prepare_df[col].isnull().sum()

    if null_count == 0:

        continue

        

    null_ratio = np.round(null_count/df_length * 100, 2)

    print("{} has {} null values ({}%)".format(col, null_count, null_ratio))
prepare_df["is_thumbnail_setted"] = 1 - prepare_df.thumbnail_url.isnull()

prepare_df.drop('thumbnail_url', axis=1, inplace=True)

prepare_df.host_response_rate = prepare_df.host_response_rate.fillna('0%').map(lambda x: float(x[:-1]))

prepare_df.host_acceptance_rate = prepare_df.host_acceptance_rate.fillna('0%').map(lambda x: float(x[:-1]))

prepare_df.bathrooms.fillna(0, inplace=True)

prepare_df.bedrooms.fillna(0, inplace=True)

prepare_df.beds.fillna(0, inplace=True)

prepare_df.cleaning_fee.fillna('$0', inplace=True)

prepare_df.review_scores_accuracy.fillna(0, inplace=True)

prepare_df.review_scores_cleanliness.fillna(0, inplace=True)

prepare_df.review_scores_checkin.fillna(0, inplace=True)

prepare_df.review_scores_communication.fillna(0, inplace=True)

prepare_df.review_scores_location.fillna(0, inplace=True)
for col in prepare_df.columns:

    if prepare_df[col].dtypes == 'object':

        print(col)
prepare_df.price = prepare_df.price.map(lambda x: float(x[1:].replace(',', '')))

prepare_df.cleaning_fee = prepare_df.cleaning_fee.map(lambda x: float(x[1:].replace(',', '')))

prepare_df.extra_people = prepare_df.extra_people.map(lambda x: float(x[1:].replace(',', '')))
X = prepare_df.drop(['id', 'easily_accomodated'], axis=1)

y = prepare_df.easily_accomodated.values



rf = RandomForestRegressor(n_estimators=100, max_depth=5)

scores = cross_val_score(rf, X, y, cv=5)
scores
rf.fit(X, y)

predictions = rf.predict(X)
plt.figure(figsize=(8, 8))



plt.plot((0, 4), (0, 4), color='gray')

plt.plot(y, predictions, linewidth=0, marker='o', alpha=0.5)

plt.grid()

plt.xlim((-0.2, 4.2))

plt.ylim((-0.2, 4.2))

plt.xlabel("Actual values")

plt.ylabel("Predicted values")

plt.show()
X = prepare_df.drop(['id', 'easily_accomodated'], axis=1)

y = np.log(prepare_df.easily_accomodated.values)



rf = RandomForestRegressor(n_estimators=100, max_depth=5)

scores = cross_val_score(rf, X, y, cv=5)

print(scores)
rf.fit(X, y)

predictions = rf.predict(X)
plt.figure(figsize=(8, 8))



plt.plot((-10, 10), (-10, 10), color='gray')

plt.plot(y, predictions, linewidth=0, marker='o', alpha=0.5)

plt.grid()

plt.xlim((-8.2, 2))

plt.ylim((-8.2, 2))

plt.xlabel("Actual values")

plt.ylabel("Predicted values")

plt.show()
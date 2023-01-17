# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

listings = pd.read_csv("/kaggle/input/berlin-airbnb-data/listings_summary.csv")

print(listings.shape)

listings.head()
# 2.1 Check column names

listings.columns
# 2.2 Check dataset info

listings.info()
# 2.3 Examie missing values

listings_na = listings.isna().sum()

listings_na[listings_na.values > 0].sort_values(ascending=False) # Find out all variables that contain missing values
# 3.1 Describe column "price"

listings.describe(include="all")["price"]
# 3.2 Convert column "price" into a numeric variable

listings["price"] = listings["price"].apply(lambda x: x.replace("$", "")) # Remove dollar sign

listings["price"] = listings["price"].apply(lambda x: x.replace(",", "")) # Remove thousand seperator

listings["price"] = listings["price"].astype("float") # Cast the column into type float

listings.describe()["price"]
# 3.3 Check outliers

import numpy as np

print("99.5% properties have a price lower than {0: .2f}".format(np.percentile(listings["price"], 99.5)))

listings = listings[(listings.price <= np.percentile(listings["price"], 99.5)) & (listings.price > 0)] # Exclude outliers
# 3.4 Create column price range

import matplotlib.pyplot as plt

plt.style.use("seaborn")

price_range = pd.cut(listings["price"], 

                     bins=[0, 20, 40, 60, 80, 100, 120, 140, listings["price"].max()], 

                     labels=["0-20", "20-40", "40-60", "60-80", "80-100", "100-120", "120-140", "140+"])

listings["price_range"] = price_range 

listings["price_range"].value_counts().sort_index().plot(kind="bar")

plt.title("Number of Listings in each Price Range")

plt.show()
## Step 4: Split listing properties

selected = []

host = ['host_is_superhost', 'host_neighbourhood', 'host_listings_count', 'host_total_listings_count', 'host_verifications', 'host_identity_verified']

location = ['neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed']

geo = ["latitude", "longitude"]

condition = ['property_type', 'room_type', 'bed_type', 'amenities', 'cleaning_fee', 'minimum_nights']

review = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']

size = ['space', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'square_feet']

others = ['instant_bookable', 'is_business_travel_ready', 'cancellation_policy']
## Step 5: Host statistics

import matplotlib.pyplot as plt

# 5.1 Descriptive statistics

listings.describe(include="all")[host]

for col in host:

    if listings[col].nunique() <= 10:

        avg_price_host = listings.groupby(col).mean()["price"]

        avg_price_host.plot(kind="bar")

        plt.title("Avg. Price grouped by "+col)

        plt.show()

    else:

        continue
# 5.2 Fill out missing values

listings["host_is_superhost"] = listings["host_is_superhost"].replace(np.NAN, "f")

listings["host_identity_verified"] = listings["host_identity_verified"].replace(np.NAN, "f")
# 5.3 Statistical test

from scipy import stats

from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

# Define multicomp function

def multicomp(target_name, group_name, data):

    if (np.nan in data[target_name]) | (np.nan in data[group_name]):

        print("Please remove NaN in target variable or group variable!")

    elif (data[target_name].nunique() == 1) | (data[group_name].nunique() == 1):

        print("There is only one unique value in target variable or group variable.")

    elif data[group_name].nunique() == 2:

        mod = MultiComparison(data[target_name], data[group_name])

        comp = mod.allpairtest(stats.ttest_ind)

        print(comp[0])

    else:

        mod = MultiComparison(data[target_name], data[group_name])

        print(mod.tukeyhsd().summary())

multicomp("price", "host_is_superhost", listings)

multicomp("price", "host_identity_verified", listings)
selected.append("host_is_superhost")

selected.append("host_identity_verified")

selected
# 5.3 Handle host verification

listings["host_ver_types"] = listings["host_verifications"].apply(lambda x: x[1:-1].replace("\'", "").split(", "))

listings["host_ver_type_counts"] = listings["host_ver_types"].apply(lambda x: len(x))

listings["host_ver_type_counts"].hist()

host_ver_types = []

for i in listings["host_ver_types"]:

    host_ver_types += i

host_ver_types_freq = dict((x, host_ver_types.count(x)) for x in set(host_ver_types))

host_ver_types_freq = pd.DataFrame.from_dict(host_ver_types_freq, orient="index")

host_ver_types_freq.reset_index(inplace=True)

host_ver_types_freq.columns = ["Verification", "Frequency"]

host_ver_types_freq = host_ver_types_freq.sort_values(by="Frequency", ascending=True)

host_ver_types_freq.plot.barh(x="Verification", y="Frequency")

plt.title("Most frequently used verification types")

plt.show()
## Step 6: Geoplot

import plotly.graph_objects as go

import plotly.express as px

from plotly.offline import plot as plotoffline

import seaborn as sns

# 6.1 Create dataset

geo = listings[['latitude', 'longitude', 'price', 'price_range']]

geo = geo.sort_values("price", ascending=True) # This sorting is necessary for the color scale to work properly. 

geo.describe()

# 6.2 Simple scatter plot

sns.scatterplot(x="longitude", 

                y="latitude", 

                hue="price", 

                data=geo, 

                alpha=0.4)
# 6.3 Map plot

px.set_mapbox_access_token("XXX") # Replace XXX with your Mapbox Token

fig = px.scatter_mapbox(geo, 

                        lat="latitude", 

                        lon="longitude", 

                        color="price_range",

                        color_discrete_sequence=px.colors.sequential.Plasma,

                        opacity=0.3, 

                        zoom=10)

fig.show()
# 6.4 Calcuate the distance bwteen the listing and mianat tractions in Berlin

# Formula to calculate distances

from math import sin, cos, sqrt, atan2, radians

def distance(lat1, lat2, lon1, lon2):

    R = 6373.0

    rlat1 = radians(lat1)

    rlat2 = radians(lat2)

    rlon1 = radians(lon1)

    rlon2 = radians(lon2)

    rdlon = rlon2 - rlon1

    rdlat = rlat2 - rlat1

    a = sin(rdlat / 2)**2 + cos(rlat1) * cos(rlat2) * sin(rdlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

# Top locations in Berlin

toploc = {"hbf": [52.525293, 13.369359], 

          "txl": [52.558794, 13.288437], 

          "btor": [52.516497, 13.377683], 

          "museum": [52.517693, 13.402141], 

          "reichstag": [52.518770, 13.376166]}

toploc = pd.DataFrame.from_dict(toploc)

toploc_trans = toploc.transpose()

toploc_trans.columns = ["latitude", "longitude"]

fig = px.scatter_mapbox(toploc_trans, 

                        lat="latitude", 

                        lon="longitude", 

                        zoom=10)

fig.show()

# Construct distance columns

dist = []

for col in toploc.columns:

    listings["dist_"+col] = listings.apply(lambda x: distance(x.latitude, toploc[col][0], x.longitude, toploc[col][1]), axis=1)

    dist.append("dist_"+col)
for distance in dist:

    sns.scatterplot(x=distance, y="price", data=listings, alpha=0.3)

    plt.title("Correlation between price and "+distance)

    plt.show()

    print("The correlation between price and "+distance+ " is {0[0]: .4f} with a p-value of {0[1]: .4f}.".format(stats.pearsonr(listings[-listings[distance].isna()][distance], 

                                                                                            listings[-listings[distance].isna()]["price"])))
for col in dist:

   listings[col+"_close"] = (listings[col] < listings[col].median())

   print(listings.groupby(col+"_close").mean()["price"])

listings["good_distance"] = listings.apply(lambda x: any([x.dist_hbf_close, x.dist_txl_close, x.dist_museum_close, x.dist_reichstag_close]), axis=1)

listings.groupby("good_distance").mean()["price"].plot(kind="bar")

plt.show()
selected.append("good_distance")

selected
## Step 7: Neighbourhood statistics

# 7.1 Top popular nerghbourhoods

neighbourhood_group_pop = pd.DataFrame(listings["neighbourhood_group_cleansed"].value_counts())

# 7.2 Average price of each neighbourhood

neighbourhood_group_price = listings.groupby("neighbourhood_group_cleansed").mean()["price"]

neighbourhood_group_price = pd.DataFrame(neighbourhood_group_price)

# 7.3 Create neighbourhood stats

neighbourhood_stat = pd.merge(neighbourhood_group_pop, 

                              neighbourhood_group_price, 

                              how="inner", left_index=True, right_index=True)

neighbourhood_stat.reset_index(inplace=True)

neighbourhood_stat.columns = ["neighbourhood_group_cleansed", "count_properties", "avg_price"]

neighbourhood_stat = neighbourhood_stat.sort_values(by="count_properties", ascending=False)

neighbourhood_stat
# 7.4 Plot

fig = plt.figure(figsize=(5, 5))

ax = neighbourhood_stat.plot(x="neighbourhood_group_cleansed", y="count_properties", kind="bar")

neighbourhood_stat.plot(x="neighbourhood_group_cleansed", y="avg_price", secondary_y=True, color="red", ax=ax)

plt.show()
listings[condition].head()
# 8.1 Property type

prop_type_avg_price = listings.groupby("property_type").mean()["price"]

prop_type_count_listings = listings["property_type"].value_counts()

prop_type_stat = pd.merge(prop_type_count_listings, prop_type_avg_price, how="inner", left_index=True, right_index=True)

prop_type_stat.columns = ["count_prop", "avg_price"]

prop_type_stat.sort_values(by="count_prop", ascending=False).head(10)
# 8.2 Room type

room_type_avg_price = listings.groupby("room_type").mean()["price"]

room_type_count_listings = listings["room_type"].value_counts()

room_type_stat = pd.merge(room_type_count_listings, room_type_avg_price, how="inner", left_index=True, right_index=True)

room_type_stat.columns = ["count_prop", "avg_price"]

room_type_stat.sort_values(by="count_prop", ascending=False).head(10)

room_type_avg_price.plot(kind="bar")

plt.title("Avg. Price per Room Type")

plt.show()
listings["is_entire_apt"] = listings["room_type"]=="Entire home/apt"

selected.append("is_entire_apt")
# 8.3 Bed type

listings["bed_type"].value_counts()
# 8.4 Amendities

listings["amenities"].head()

listings["amenities"] = listings["amenities"].apply(lambda x: x[1:-1].replace("\'", "").split(","))
listings["amenities"].head()

amenity_types = []

for i in listings["amenities"]:

    amenity_types += i

amenity_types_freq = dict((x, amenity_types.count(x)) for x in set(amenity_types))

amenity_types_freq = pd.DataFrame.from_dict(amenity_types_freq, orient="index")

amenity_types_freq.reset_index(inplace=True)

amenity_types_freq.columns = ["Amenity", "Frequency"]

amenity_types_freq = amenity_types_freq.sort_values(by="Frequency", ascending=False)

amenity_types_freq.head(20).plot.barh(x="Amenity", y="Frequency")

plt.title("Top20 most frequent amenity types")

plt.show()
listings["with_hair_dryer"] = listings["amenities"].apply(lambda x: '"Hair dryer"' in x)

listings["lap_friendly"] = listings["amenities"].apply(lambda x: '"Laptop friendly workspace"' in x)

listings["with_hanger"] = listings["amenities"].apply(lambda x: "Hangers" in x)

print(multicomp("price", "with_hair_dryer", listings))

print(multicomp("price", "lap_friendly", listings))

print(multicomp("price", "with_hanger", listings))

for i in ["with_hair_dryer", "lap_friendly", "with_hanger"]:

    selected.append(i)
# 8.5 Minimum nights

listings["minimum_nights"].describe()

listings["min_nights_greater_than_two"] = listings["minimum_nights"] > 2

multicomp("price", "min_nights_greater_than_two", data=listings)
selected.append("min_nights_greater_than_two")
# 8.6 Cleaning fee

# Remove dollar sign

listings["cleaning_fee"][-listings["cleaning_fee"].isna()] = listings["cleaning_fee"][-listings["cleaning_fee"].isna()].apply(lambda x: x.replace("$", "").replace(",", ""))

listings["cleaning_fee"] = listings["cleaning_fee"].astype("float")
listings["cleaning_fee"].isna().sum() # Check missing values

listings["cleaning_fee"].describe()

sns.scatterplot(x="cleaning_fee", y="price", data=listings, alpha=0.3)

plt.title("Correlation bewteen Cleaning Fee and Price")

plt.show()

print("The correlation between cleaning fee and price is {0[0]: .4f} with a p-value of {0[1]: .4f}.".format(stats.pearsonr(listings[-listings["cleaning_fee"].isna()]["cleaning_fee"], 

                                                                                            listings[-listings["cleaning_fee"].isna()]["price"])))
selected.append("cleaning_fee")

selected
# Step 9: Review statistics

# 9.1 Examine the distribution of score ratings

listings["review_scores_rating"].hist()
# 9.2 Scatter plot between review score and price

import seaborn as sns

import scipy.stats as stats

sns.regplot(x="review_scores_rating", y="price", data=listings[listings["review_scores_rating"]>=75])

plt.title("Price vs Review Score Rating")

plt.show()

print("The correlation between review score and price is {0[0]: .4f} with a p-value of {0[1]: .4f}.".format(stats.pearsonr(listings[-listings["review_scores_rating"].isna()]["review_scores_rating"], 

                                                                                            listings[-listings["review_scores_rating"].isna()]["price"])))
# 9.3 Check the correlation between price and other scores

for col in review:

    print(("The pearson correlation coefficient between " + col + " and price is {0[0]: .4f}.").format(stats.pearsonr(listings[-listings[col].isna()][col], 

                                                                                            listings[-listings[col].isna()]["price"])))
# Step 10: Size

# 10.1 Look at size-related variables

listings[size].head(10)
# 10.2 Check the correlation between number of accommodates and price

listings["accommodates"].hist()

listings["accommodates"].describe()
listings.groupby("accommodates").mean()["price"].plot(kind="bar")

plt.title("Avg. Price grouped by Number of Accommodates")

plt.show()

print("The pearson correlation coefficient between ther number of acoommodates and price is {0[0]: .4f} with a p-value of {0[1]: .4f}.".format(stats.pearsonr(listings["accommodates"], listings["price"])))
selected.append("accommodates")

selected
# 10.3 Check the correlation bewteen accommodates and other size variables

size_variables = listings[size]

size_variables.drop(["space", "square_feet"], axis=1, inplace=True)

size_variables.head()
size_corr = size_variables.corr()

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(size_corr, cmap=colormap, annot=True, fmt=".4f")
# Step 11: Other conditions

listings[others].head()
# 11.1 Instant bookable

listings["instant_bookable"].value_counts()

multicomp("price", "instant_bookable", listings)

listings.groupby("instant_bookable").mean()["price"].plot(kind="bar")

plt.title("Avg. Price split by Instant Bookable Policy")

plt.show()

selected.append("instant_bookable")
# 11.2 Ready for Business Travel

listings["is_business_travel_ready"].value_counts()

multicomp("price", "is_business_travel_ready", listings)

listings.groupby("is_business_travel_ready").mean()["price"].plot(kind="bar")

plt.title("Avg. Price split by Ready for Business Travel")

plt.show()
# 11.3 Cancellation plicy

print(listings["cancellation_policy"].value_counts())

multicomp("price", "cancellation_policy", listings)

listings.groupby("cancellation_policy").mean()["price"].plot(kind="bar")

plt.title("Avg. Price split by Cancellation Policy")

plt.show()
listings["cancellation_non_flexible"] = listings["cancellation_policy"]!="flexible"

listings["cancellation_non_flexible"].value_counts()

multicomp("price", "cancellation_non_flexible", listings)

selected.append("cancellation_non_flexible")
listings[selected].info()
# 1.1 Convert string variables into categorical variables

listings["host_is_superhost"] = listings["host_is_superhost"]=="t"

listings["host_identity_verified"] = listings["host_identity_verified"]=="t"

listings["instant_bookable"] = listings["instant_bookable"]=="t"
for col in listings[selected].select_dtypes("bool").columns:

    listings[col] = listings[col].astype("int")
listings[selected].info()
# 1.2 Standardisation

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

scaledFeatures = sc.fit_transform(listings[selected])
# 1.3 Load packages and create test set

import xgboost as xgb

from xgboost import plot_importance

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score, mean_squared_error



X = scaledFeatures

y = listings["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 2.1 Initialize XGBoost classifier and find the best parameter sets with Grid Search CV

xgb_clf = xgb.XGBRegressor()

parameters = {'n_estimators': [120, 100, 140], 'max_depth':[3,5,7,9]}

grid_search = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
# 2.2 Xgb with best parameters

xgb_clf = xgb.XGBRegressor(n_estimators=100, max_depth=5)

xgb_clf.fit(X_train, y_train)

y_test_pred = xgb_clf.predict(X_test)

print("R^2 score is: {0: .4f}".format(r2_score(y_test, y_test_pred)))

print("RMSE is: {0: .4f}".format(np.sqrt(mean_squared_error(y_test, y_test_pred))))
# 2.3 Plot feature importance

featureImport = pd.DataFrame(xgb_clf.feature_importances_, index=selected)

featureImport.columns = ["Importance"]

featureImport.sort_values(["Importance"], ascending=True).plot(kind="barh")

plt.title("XGBoost Relative Feature Importance")

plt.show()